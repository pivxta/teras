use super::MAX_PLY;
use arrayvec::ArrayVec;
use dama::Move;

#[derive(Clone, Debug)]
pub struct Line(ArrayVec<Move, { Self::MAX_LEN }>);

impl Line {
    pub const MAX_LEN: usize = MAX_PLY as usize;

    pub fn new() -> Self {
        Line(ArrayVec::new())
    }

    pub fn from_move(mv: Move) -> Self {
        Line(ArrayVec::from_iter([mv]))
    }

    pub fn clear(&mut self) {
        self.0.clear();
    }

    pub fn set(&mut self, mv: Move, sub_line: &Line) {
        self.0.clear();
        let _ = self.0.try_push(mv);
        let _ = self.0.try_extend_from_slice(&sub_line.0);
    }

    pub fn first(&self) -> Option<Move> {
        self.0.first().cloned()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn to_vec(&self) -> Vec<Move> {
        self.0.to_vec()
    }
}
