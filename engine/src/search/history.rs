use super::Node;
use dama::{ByColor, ByPiece, BySquare, Color, Move, Piece, Position, Square};

pub const HISTORY_LIMIT: i32 = 8192;
pub const HISTORY_BONUS_MULTIPLIER: i32 = 8;
pub const HISTORY_AGE_DIVISOR: i32 = 128;

#[derive(Clone, Debug, Default)]
pub struct Histories {
    history: Box<ByColor<ByPiece<BySquare<i32>>>>,
}

impl Histories {
    #[inline]
    pub fn clear(&mut self) {
        self.history = Default::default();
    }

    #[inline]
    pub fn get(&self, position: &Position, mv: &Move) -> i32 {
        let moved = position.piece_at(mv.from).expect("no piece to be moved");
        self.history[position.side_to_move()][moved][mv.to]
    }

    #[inline]
    pub fn bonus(&mut self, position: &Position, node: &Node, mv: &Move) {
        let color = position.side_to_move();
        let moved = position.piece_at(mv.from).expect("no piece to be moved");
        let value = &mut self.history[color][moved][mv.to];

        let bonus = HISTORY_BONUS_MULTIPLIER * (node.depth * node.depth) as i32;
        let bonus = bonus - bonus.saturating_mul(*value) / HISTORY_LIMIT;

        *value = (*value + bonus).min(HISTORY_LIMIT);
    }

    #[inline]
    pub fn penalty(&mut self, position: &Position, node: &Node, mv: &Move) {
        let color = position.side_to_move();
        let moved = position.piece_at(mv.from).expect("no piece to be moved");
        let value = &mut self.history[color][moved][mv.to];

        let penalty = HISTORY_BONUS_MULTIPLIER * (node.depth * node.depth) as i32;
        let penalty = penalty + penalty.saturating_mul(*value) / HISTORY_LIMIT;

        *value = (*value - penalty).max(-HISTORY_LIMIT);
    }
}
