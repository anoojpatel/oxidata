use pyo3::prelude::*;
use pyo3::types::PyByteArray;
use pyo3::exceptions::PyOSError;
use libc;
use std::sync::atomic::{AtomicI64 as StdAtomicI64, Ordering};
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, RwLock};

fn shm_name_to_posix(name: &str) -> String {
    if name.starts_with('/') {
        name.to_string()
    } else {
        format!("/{}", name)
    }
}

unsafe fn mmap_shm_ro(name: &str) -> Result<(*mut libc::c_void, usize, libc::c_int), i32> {
    let posix = shm_name_to_posix(name);
    let cstr = std::ffi::CString::new(posix).unwrap();
    let fd = libc::shm_open(cstr.as_ptr(), libc::O_RDONLY, 0);
    if fd < 0 {
        return Err(*libc::__error());
    }
    let mut st: libc::stat = std::mem::zeroed();
    if libc::fstat(fd, &mut st as *mut libc::stat) != 0 {
        let e = *libc::__error();
        libc::close(fd);
        return Err(e);
    }
    let len = st.st_size as usize;
    let ptr = libc::mmap(
        std::ptr::null_mut(),
        len,
        libc::PROT_READ,
        libc::MAP_SHARED,
        fd,
        0,
    );
    if ptr == libc::MAP_FAILED {
        let e = *libc::__error();
        libc::close(fd);
        return Err(e);
    }
    Ok((ptr, len, fd))
}

unsafe fn mmap_shm_rw(name: &str) -> Result<(*mut libc::c_void, usize, libc::c_int), i32> {
    let posix = shm_name_to_posix(name);
    let cstr = std::ffi::CString::new(posix).unwrap();
    let fd = libc::shm_open(cstr.as_ptr(), libc::O_RDWR, 0);
    if fd < 0 {
        return Err(*libc::__error());
    }
    let mut st: libc::stat = std::mem::zeroed();
    if libc::fstat(fd, &mut st as *mut libc::stat) != 0 {
        let e = *libc::__error();
        libc::close(fd);
        return Err(e);
    }
    let len = st.st_size as usize;
    let ptr = libc::mmap(
        std::ptr::null_mut(),
        len,
        libc::PROT_READ | libc::PROT_WRITE,
        libc::MAP_SHARED,
        fd,
        0,
    );
    if ptr == libc::MAP_FAILED {
        let e = *libc::__error();
        libc::close(fd);
        return Err(e);
    }
    Ok((ptr, len, fd))
}

unsafe fn munmap_close(ptr: *mut libc::c_void, len: usize, fd: libc::c_int) {
    libc::munmap(ptr, len);
    libc::close(fd);
}

fn errno() -> i32 {
    unsafe { *libc::__error() }
}

#[repr(C)]
struct RingHeader {
    head: AtomicU64,
    tail: AtomicU64,
    capacity: u64,
    slot_size: u64,
}

fn header_size() -> usize {
    std::mem::size_of::<RingHeader>()
}

unsafe fn mmap_ring_create(name: &str, capacity: u64, slot_size: u64) -> Result<(*mut u8, usize, libc::c_int), i32> {
    let posix = shm_name_to_posix(name);
    let cstr = std::ffi::CString::new(posix).unwrap();
    let fd = libc::shm_open(cstr.as_ptr(), libc::O_CREAT | libc::O_EXCL | libc::O_RDWR, 0o600);
    if fd < 0 {
        return Err(errno());
    }

    let total = header_size() as u64 + capacity.saturating_mul(slot_size);
    if libc::ftruncate(fd, total as libc::off_t) != 0 {
        let e = errno();
        libc::close(fd);
        return Err(e);
    }

    let ptr = libc::mmap(
        std::ptr::null_mut(),
        total as usize,
        libc::PROT_READ | libc::PROT_WRITE,
        libc::MAP_SHARED,
        fd,
        0,
    );
    if ptr == libc::MAP_FAILED {
        let e = errno();
        libc::close(fd);
        return Err(e);
    }

    let base = ptr as *mut u8;
    let hdr = base as *mut RingHeader;
    std::ptr::write(
        hdr,
        RingHeader {
            head: AtomicU64::new(0),
            tail: AtomicU64::new(0),
            capacity,
            slot_size,
        },
    );

    Ok((base, total as usize, fd))
}

unsafe fn mmap_ring_attach(name: &str) -> Result<(*mut u8, usize, libc::c_int), i32> {
    let posix = shm_name_to_posix(name);
    let cstr = std::ffi::CString::new(posix).unwrap();
    let fd = libc::shm_open(cstr.as_ptr(), libc::O_RDWR, 0);
    if fd < 0 {
        return Err(errno());
    }

    let mut st: libc::stat = std::mem::zeroed();
    if libc::fstat(fd, &mut st as *mut libc::stat) != 0 {
        let e = errno();
        libc::close(fd);
        return Err(e);
    }
    let len = st.st_size as usize;

    let ptr = libc::mmap(
        std::ptr::null_mut(),
        len,
        libc::PROT_READ | libc::PROT_WRITE,
        libc::MAP_SHARED,
        fd,
        0,
    );
    if ptr == libc::MAP_FAILED {
        let e = errno();
        libc::close(fd);
        return Err(e);
    }

    Ok((ptr as *mut u8, len, fd))
}

#[pyclass]
struct ShmRingBuffer {
    name: String,
    base: *mut u8,
    len: usize,
    fd: libc::c_int,
}

unsafe impl Send for ShmRingBuffer {}
unsafe impl Sync for ShmRingBuffer {}

#[pymethods]
impl ShmRingBuffer {
    #[new]
    fn new(name: String, capacity: u64, slot_size: u64, create: bool) -> PyResult<Self> {
        let res = unsafe {
            if create {
                mmap_ring_create(&name, capacity, slot_size)
            } else {
                mmap_ring_attach(&name)
            }
        };

        match res {
            Ok((base, len, fd)) => Ok(Self { name, base, len, fd }),
            Err(e) => Err(PyOSError::new_err(format!("ShmRingBuffer open failed: errno {}", e))),
        }
    }

    fn name(&self) -> String {
        self.name.clone()
    }

    fn capacity(&self) -> u64 {
        unsafe { (*(self.base as *const RingHeader)).capacity }
    }

    fn slot_size(&self) -> u64 {
        unsafe { (*(self.base as *const RingHeader)).slot_size }
    }

    fn push(&self, py: Python<'_>, data: &[u8]) -> PyResult<bool> {
        let ok = py.allow_threads(|| unsafe {
            let hdr = &*(self.base as *const RingHeader);
            let cap = hdr.capacity;
            let slot = hdr.slot_size as usize;
            if slot < 8 {
                return false;
            }
            let max_payload = slot - 8;
            let n = std::cmp::min(data.len(), max_payload);

            let head = hdr.head.load(Ordering::Acquire);
            let tail = hdr.tail.load(Ordering::Acquire);
            if tail.wrapping_sub(head) >= cap {
                return false;
            }

            let idx = (tail % cap) as usize;
            let slot_off = header_size() + idx * slot;
            let p = self.base.add(slot_off);

            std::ptr::write_unaligned(p as *mut u64, n as u64);
            std::ptr::copy_nonoverlapping(data.as_ptr(), p.add(8), n);
            hdr.tail.store(tail.wrapping_add(1), Ordering::Release);
            true
        });

        Ok(ok)
    }

    fn push_handle(&self, py: Python<'_>, offset: u64, nbytes: u64, kind_tag: u32) -> PyResult<bool> {
        let mut buf = [0u8; 24];
        buf[0..8].copy_from_slice(&offset.to_le_bytes());
        buf[8..16].copy_from_slice(&nbytes.to_le_bytes());
        buf[16..20].copy_from_slice(&kind_tag.to_le_bytes());
        // last 4 bytes reserved
        self.push(py, &buf)
    }

    fn pop_handle(&self, py: Python<'_>) -> PyResult<Option<(u64, u64, u32)>> {
        let out = PyByteArray::new_bound(py, &[0u8; 24]);
        let n = self.pop_into(py, &out, 0)?;
        if n == 0 {
            return Ok(None);
        }
        if n < 20 {
            return Ok(None);
        }

        let bytes = unsafe { &out.as_bytes()[..n] };
        let mut a = [0u8; 8];
        a.copy_from_slice(&bytes[0..8]);
        let offset = u64::from_le_bytes(a);
        a.copy_from_slice(&bytes[8..16]);
        let nbytes = u64::from_le_bytes(a);
        let mut b = [0u8; 4];
        b.copy_from_slice(&bytes[16..20]);
        let kind_tag = u32::from_le_bytes(b);
        Ok(Some((offset, nbytes, kind_tag)))
    }

    fn pop_into(&self, py: Python<'_>, out: &Bound<'_, PyByteArray>, out_offset: usize) -> PyResult<usize> {
        let out_len = out.len();
        if out_offset > out_len {
            return Ok(0);
        }
        let dst = unsafe { out.data().add(out_offset) as usize };

        let copied = py.allow_threads(|| unsafe {
            let hdr = &*(self.base as *const RingHeader);
            let cap = hdr.capacity;
            let slot = hdr.slot_size as usize;

            let head = hdr.head.load(Ordering::Acquire);
            let tail = hdr.tail.load(Ordering::Acquire);
            if head == tail {
                return 0usize;
            }

            let idx = (head % cap) as usize;
            let slot_off = header_size() + idx * slot;
            let p = self.base.add(slot_off);
            let n = std::ptr::read_unaligned(p as *const u64) as usize;

            let max_out = out_len - out_offset;
            let take = std::cmp::min(n, max_out);
            let src = p.add(8);
            std::ptr::copy_nonoverlapping(src, dst as *mut u8, take);
            hdr.head.store(head.wrapping_add(1), Ordering::Release);
            take
        });

        Ok(copied)
    }

    fn close(&mut self) {
        unsafe {
            munmap_close(self.base as *mut libc::c_void, self.len, self.fd);
        }
        self.base = std::ptr::null_mut();
        self.len = 0;
        self.fd = -1;
    }

    fn unlink(&self) -> PyResult<()> {
        let posix = shm_name_to_posix(&self.name);
        let cstr = std::ffi::CString::new(posix).unwrap();
        let rc = unsafe { libc::shm_unlink(cstr.as_ptr()) };
        if rc != 0 {
            return Err(PyOSError::new_err(format!("shm_unlink failed: errno {}", errno())));
        }
        Ok(())
    }
}

#[pyclass]
struct AtomicI64 {
    inner: Arc<StdAtomicI64>,
}

#[pymethods]
impl AtomicI64 {
    #[new]
    fn new(value: i64) -> Self {
        Self {
            inner: Arc::new(StdAtomicI64::new(value)),
        }
    }

    fn load(&self) -> i64 {
        Python::with_gil(|py| py.allow_threads(|| self.inner.load(Ordering::SeqCst)))
    }

    fn store(&self, value: i64) {
        Python::with_gil(|py| py.allow_threads(|| self.inner.store(value, Ordering::SeqCst)))
    }

    fn fetch_add(&self, delta: i64) -> i64 {
        Python::with_gil(|py| py.allow_threads(|| self.inner.fetch_add(delta, Ordering::SeqCst)))
    }
}

#[pyclass]
struct RwBytes {
    inner: Arc<RwLock<Vec<u8>>>,
}

#[pymethods]
impl RwBytes {
    #[new]
    fn new(size: usize) -> Self {
        Self {
            inner: Arc::new(RwLock::new(vec![0u8; size])),
        }
    }

    fn size(&self) -> usize {
        Python::with_gil(|py| py.allow_threads(|| self.inner.read().unwrap().len()))
    }

    fn readinto(&self, py: Python<'_>, out: &Bound<'_, PyByteArray>, offset: usize) -> PyResult<usize> {
        let out_len = out.len();
        let dst = out.data() as usize;
        let copied = py.allow_threads(|| {
            let guard = self.inner.read().unwrap();
            if offset >= guard.len() {
                return 0usize;
            }
            let max_n = guard.len() - offset;
            let n = std::cmp::min(out_len, max_n);
            let src = &guard[offset..offset + n];
            unsafe {
                std::ptr::copy_nonoverlapping(src.as_ptr(), dst as *mut u8, n);
            }
            n
        });
        Ok(copied)
    }

    fn write(&self, py: Python<'_>, data: &[u8], offset: usize) -> PyResult<usize> {
        let copied = py.allow_threads(|| {
            let mut guard = self.inner.write().unwrap();
            if offset >= guard.len() {
                return 0usize;
            }
            let max_n = guard.len() - offset;
            let n = std::cmp::min(data.len(), max_n);
            guard[offset..offset + n].copy_from_slice(&data[..n]);
            n
        });
        Ok(copied)
    }
}

#[pymodule]
fn oxidata_native(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<AtomicI64>()?;
    m.add_class::<RwBytes>()?;
    m.add_class::<ShmRingBuffer>()?;

    #[pyfn(m)]
    fn shm_readinto(
        py: Python<'_>,
        shm_name: &str,
        offset: usize,
        out: &Bound<'_, PyByteArray>,
        out_offset: usize,
        nbytes: Option<usize>,
    ) -> PyResult<usize> {
        let out_len = out.len();
        if out_offset > out_len {
            return Ok(0);
        }
        let max_out = out_len - out_offset;
        let want = nbytes.unwrap_or(max_out);
        let want = std::cmp::min(want, max_out);
        let dst = unsafe { out.data().add(out_offset) as usize };

        let copied = py.allow_threads(|| unsafe {
            let (ptr, len, fd) = mmap_shm_ro(shm_name).map_err(|e| e)?;
            let res = if offset >= len {
                0usize
            } else {
                let max_n = len - offset;
                let n = std::cmp::min(want, max_n);
                let src = (ptr as *const u8).add(offset);
                std::ptr::copy_nonoverlapping(src, dst as *mut u8, n);
                n
            };
            munmap_close(ptr, len, fd);
            Ok::<usize, i32>(res)
        });

        match copied {
            Ok(n) => Ok(n),
            Err(e) => Err(PyOSError::new_err(format!("shm_readinto failed: errno {}", e))),
        }
    }

    #[pyfn(m)]
    fn shm_write(
        py: Python<'_>,
        shm_name: &str,
        offset: usize,
        data: &[u8],
    ) -> PyResult<usize> {
        let wrote = py.allow_threads(|| unsafe {
            let (ptr, len, fd) = mmap_shm_rw(shm_name).map_err(|e| e)?;
            let res = if offset >= len {
                0usize
            } else {
                let max_n = len - offset;
                let n = std::cmp::min(data.len(), max_n);
                let dst = (ptr as *mut u8).add(offset);
                std::ptr::copy_nonoverlapping(data.as_ptr(), dst, n);
                n
            };
            munmap_close(ptr, len, fd);
            Ok::<usize, i32>(res)
        });

        match wrote {
            Ok(n) => Ok(n),
            Err(e) => Err(PyOSError::new_err(format!("shm_write failed: errno {}", e))),
        }
    }

    Ok(())
}
