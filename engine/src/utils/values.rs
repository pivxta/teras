use dama::{ByPiece, Piece};

const PIECES: ByPiece<i32> = ByPiece {
    pawn: 100,
    knight: 300,
    bishop: 300,
    rook: 500,
    queen: 950,
    king: 10000,
};

pub fn piece(piece: Piece) -> i32 {
    PIECES[piece]
}
