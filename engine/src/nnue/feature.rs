use dama::{Color, Piece, Square};

#[inline]
pub fn feature(perspective: Color, color: Color, piece: Piece, square: Square) -> usize {
    let square = match perspective {
        Color::White => square,
        Color::Black => square.flip_vertical(),
    };
    let index = if perspective == color { 0 } else { 1 };
    let index = index * Piece::COUNT + piece as usize;
    index * Square::COUNT + square as usize
}
