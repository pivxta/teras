#[repr(align(32))]
#[derive(Copy, Clone, Default, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct SimdAlign<T>(pub T);

impl<T> std::ops::Deref for SimdAlign<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T> std::ops::DerefMut for SimdAlign<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

impl<T> AsRef<T> for SimdAlign<T> {
    #[inline]
    fn as_ref(&self) -> &T {
        &self.0
    }
}

impl<T> AsMut<T> for SimdAlign<T> {
    #[inline]
    fn as_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

impl<T> From<T> for SimdAlign<T> {
    #[inline]
    fn from(value: T) -> Self {
        Self(value)
    }
}
