use crate::eval::Eval;
use core::ops;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Window {
    pub alpha: Eval,
    pub beta: Eval,
}

impl Window {
    pub const FULL: Window = Window::new(Eval::MIN, Eval::MAX);

    #[inline]
    pub const fn new(alpha: Eval, beta: Eval) -> Self {
        Window { alpha, beta }
    }

    #[inline]
    pub fn around(middle: Eval, delta: i32) -> Self {
        Window {
            alpha: middle - delta,
            beta: middle + delta,
        }
    }

    #[inline]
    pub fn contains(&self, eval: Eval) -> bool {
        eval > self.alpha && eval < self.beta
    }

    #[inline]
    pub fn null_alpha(&mut self) -> Self {
        Self {
            alpha: self.alpha,
            beta: self.alpha + 1,
        }
    }

    #[inline]
    pub fn null_beta(&mut self) -> Self {
        Self {
            alpha: self.beta - 1,
            beta: self.beta,
        }
    }
}

impl ops::Neg for Window {
    type Output = Window;

    #[inline]
    fn neg(self) -> Window {
        Window {
            alpha: -self.beta,
            beta: -self.alpha,
        }
    }
}
