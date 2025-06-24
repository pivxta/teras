use super::Node;
use dama::{ByColor, ByPiece, BySquare, Move, Position};

pub const HISTORY_LIMIT: i32 = 8192;
pub const HISTORY_BONUS_MULTIPLIER: i32 = 8;

#[derive(Clone, Debug, Default)]
pub struct History {
    history: Box<ByColor<ByPiece<BySquare<Value>>>>,
}

impl History {
    #[inline]
    pub fn clear(&mut self) {
        self.history = Default::default();
    }

    #[inline]
    pub fn get(&self, position: &Position, mv: &Move) -> i32 {
        let moved = position.piece_at(mv.from).expect("no piece to be moved");
        self.history[position.side_to_move()][moved][mv.to].0
    }

    #[inline]
    pub fn bonus(&mut self, position: &Position, node: &Node, mv: &Move) {
        let color = position.side_to_move();
        let moved = position.piece_at(mv.from).expect("no piece to be moved");
        let value = &mut self.history[color][moved][mv.to];

        value.update(HISTORY_BONUS_MULTIPLIER * (node.depth * node.depth) as i32);
    }

    #[inline]
    pub fn penalty(&mut self, position: &Position, node: &Node, mv: &Move) {
        let color = position.side_to_move();
        let moved = position.piece_at(mv.from).expect("no piece to be moved");
        let value = &mut self.history[color][moved][mv.to];

        value.update(-HISTORY_BONUS_MULTIPLIER * (node.depth * node.depth) as i32);
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
struct Value(i32);

impl Value {
    fn update(&mut self, bonus: i32) {
        self.0 += bonus - bonus.abs() * self.0 / HISTORY_LIMIT;
        self.0 = self.0.clamp(-HISTORY_LIMIT, HISTORY_LIMIT);
    }
}
