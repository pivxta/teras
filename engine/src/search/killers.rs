use std::array;

use super::{MAX_PLY, Node};
use arrayvec::ArrayVec;
use dama::Move;

#[derive(Clone, Debug)]
pub struct Killers {
    killers: [ArrayVec<Move, 2>; MAX_PLY as usize],
}

impl Default for Killers {
    #[inline]
    fn default() -> Self {
        Self {
            killers: array::from_fn(|_| ArrayVec::new()),
        }
    }
}

impl Killers {
    pub fn clear(&mut self) {
        self.killers = array::from_fn(|_| Default::default());
    }

    #[inline]
    pub fn add(&mut self, node: &Node, mv: Move) {
        let killers = &mut self.killers[node.ply as usize];
        if killers.contains(&mv) {
            return;
        }
        if killers.len() == 2 {
            killers.pop();
        }
        killers.insert(0, mv);
    }

    #[inline]
    pub fn get(&self, node: &Node) -> &[Move] {
        &self.killers[node.ply as usize]
    }
}
