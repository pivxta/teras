use dama::{ByColor, FenError, Position, UciMove};
use std::{str::FromStr, time::Duration};
use thiserror::Error;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Command {
    Uci,
    IsReady,
    Debug(bool),
    NewGame,
    SetOption {
        name: String,
        value: Option<String>,
    },
    Position {
        position: Position,
        moves: Vec<UciMove>,
    },
    Go(Go),
    Eval,
    Stop,
    PonderHit,
    Quit,
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum CommandParseError {
    #[error("no command submitted")]
    NoCommand,
    #[error("invalid command: '{0}'")]
    InvalidCommand(String),
    #[error("invalid command usage")]
    InvalidUsage,
    #[error("invalid argument: '{0}'")]
    InvalidArgument(String),
    #[error("invalid move: {0}")]
    InvalidMove(String),
    #[error("invalid fen '{fen}': {error}")]
    InvalidFen { fen: String, error: FenError },
    #[error("expected value for parameter")]
    ExpectedValue,
}

impl FromStr for Command {
    type Err = CommandParseError;

    fn from_str(s: &str) -> Result<Command, CommandParseError> {
        let mut args = s.split_whitespace();
        match args.next().ok_or(CommandParseError::NoCommand)? {
            "uci" => Ok(Command::Uci),
            "quit" => Ok(Command::Quit),
            "isready" => Ok(Command::IsReady),
            "ucinewgame" => Ok(Command::NewGame),
            "ponderhit" => Ok(Command::PonderHit),
            "eval" => Ok(Command::Eval),
            "stop" => Ok(Command::Stop),
            "debug" => Command::parse_debug(args),
            "position" => Command::parse_position(args),
            "setoption" => Command::parse_setoption(args),
            "go" => Command::parse_go(args),
            command => Err(CommandParseError::InvalidCommand(command.to_owned())),
        }
    }
}

impl Command {
    fn parse_debug<'a, I>(mut args: I) -> Result<Command, CommandParseError>
    where
        I: Iterator<Item = &'a str>,
    {
        let value = match args.next() {
            Some("on") | None => true,
            Some("off") => false,
            Some(arg) => return Err(CommandParseError::InvalidArgument(arg.to_owned())),
        };

        if let Some(arg) = args.next() {
            return Err(CommandParseError::InvalidArgument(arg.to_owned()));
        }

        Ok(Command::Debug(value))
    }

    fn parse_position<'a, I>(args: I) -> Result<Command, CommandParseError>
    where
        I: Iterator<Item = &'a str> + Clone,
    {
        let mut args = args.peekable();
        let position = match args.next_if(|&arg| arg != "moves") {
            Some("startpos") | None => Position::new_initial(),
            Some("fen") => {
                let fen = args
                    .clone()
                    .take_while(|&arg| arg != "moves")
                    .collect::<Vec<_>>()
                    .join(" ");

                while args.next_if(|&arg| arg != "moves").is_some() {}

                Position::from_fen(&fen)
                    .map_err(|error| CommandParseError::InvalidFen { fen, error })?
            }
            Some(arg) => return Err(CommandParseError::InvalidArgument(arg.to_owned())),
        };
        let moves = Self::parse_moves(args)?;

        Ok(Command::Position { position, moves })
    }

    fn parse_moves<'a, I>(mut args: I) -> Result<Vec<UciMove>, CommandParseError>
    where
        I: Iterator<Item = &'a str> + Clone,
    {
        match args.next() {
            None => Ok(Vec::new()),
            Some("moves") => {
                let moves = args
                    .map(|s| {
                        s.parse::<UciMove>()
                            .map_err(|_| CommandParseError::InvalidMove(s.to_owned()))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(moves)
            }
            Some(arg) => Err(CommandParseError::InvalidArgument(arg.to_owned())),
        }
    }

    fn parse_setoption<'a, I>(args: I) -> Result<Command, CommandParseError>
    where
        I: Iterator<Item = &'a str> + Clone,
    {
        let mut args = args.peekable();

        let name = match args.next() {
            Some("name") => {
                let name = args
                    .clone()
                    .take_while(|&arg| arg != "value")
                    .collect::<Vec<_>>()
                    .join(" ");

                while args.next_if(|&arg| arg != "value").is_some() {}

                name
            }
            Some(arg) => return Err(CommandParseError::InvalidArgument(arg.to_owned())),
            None => return Err(CommandParseError::InvalidUsage),
        };
        let value = args.next().map(|_| args.collect::<Vec<_>>().join(" "));

        Ok(Command::SetOption { name, value })
    }

    fn parse_go<'a, I>(args: I) -> Result<Command, CommandParseError>
    where
        I: Iterator<Item = &'a str> + Clone,
    {
        let mut args = args.peekable();
        let mut limits = Go::default();

        while let Some(param) = args.next() {
            match param {
                "infinite" => {}
                "ponder" => limits.ponder = true,
                "depth" => limits.depth = Some(Self::parse_value(args.next())?),
                "nodes" => limits.nodes = Some(Self::parse_value(args.next())?),
                "mate" => limits.mate = Some(Self::parse_value(args.next())?),
                "wtime" => {
                    limits.time.white = Some(Duration::from_millis(Self::parse_value(args.next())?))
                }
                "btime" => {
                    limits.time.black = Some(Duration::from_millis(Self::parse_value(args.next())?))
                }
                "winc" => {
                    limits.increment.white = Duration::from_millis(Self::parse_value(args.next())?)
                }
                "binc" => {
                    limits.increment.black = Duration::from_millis(Self::parse_value(args.next())?)
                }
                "movestogo" => limits.moves_to_go = Some(Self::parse_value::<u32>(args.next())?),
                "movetime" => {
                    limits.move_time = Some(Duration::from_millis(Self::parse_value(args.next())?))
                }
                "searchmoves" => {
                    while let Some(mv) = args.peek() {
                        if let Ok(mv) = mv.parse::<UciMove>() {
                            limits.moves_to_search.push(mv);
                            args.next();
                        } else {
                            break;
                        }
                    }
                }
                arg => return Err(CommandParseError::InvalidArgument(arg.to_owned())),
            }
        }

        Ok(Command::Go(limits))
    }

    fn parse_value<T: FromStr>(arg: Option<&str>) -> Result<T, CommandParseError> {
        let arg = arg.ok_or(CommandParseError::ExpectedValue)?;
        arg.parse::<T>()
            .map_err(|_| CommandParseError::InvalidArgument(arg.to_owned()))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct Go {
    pub moves_to_search: Vec<UciMove>,
    pub depth: Option<u32>,
    pub mate: Option<u32>,
    pub nodes: Option<u64>,
    pub move_time: Option<Duration>,
    pub time: ByColor<Option<Duration>>,
    pub increment: ByColor<Duration>,
    pub moves_to_go: Option<u32>,
    pub ponder: bool,
}

#[cfg(test)]
mod tests {
    use crate::command::Go;
    use dama::ByColor;

    use super::{Command, CommandParseError};
    use std::time::Duration;

    #[test]
    fn parse_simple() {
        assert_eq!("uci".parse::<Command>(), Ok(Command::Uci));
        assert_eq!("isready".parse::<Command>(), Ok(Command::IsReady));
        assert_eq!("ucinewgame".parse::<Command>(), Ok(Command::NewGame));
        assert_eq!("debug".parse::<Command>(), Ok(Command::Debug(true)));
        assert_eq!("debug  on".parse::<Command>(), Ok(Command::Debug(true)));
        assert_eq!("debug off".parse::<Command>(), Ok(Command::Debug(false)));
        assert_eq!(
            "debug off something".parse::<Command>(),
            Err(CommandParseError::InvalidArgument("something".into()))
        );
    }

    #[test]
    fn parse_setoption() {
        assert_eq!(
            "setoption name Skill Level value 10".parse::<Command>(),
            Ok(Command::SetOption {
                name: "Skill Level".into(),
                value: Some("10".into())
            })
        );
        assert_eq!(
            "setoption name Clear Hash".parse::<Command>(),
            Ok(Command::SetOption {
                name: "Clear Hash".into(),
                value: None
            })
        );
    }

    #[test]
    fn parse_go() {
        assert_eq!(
            "go infinite".parse::<Command>(),
            Ok(Command::Go(Go::default()))
        );
        assert_eq!(
            "go movetime 100 depth 7 nodes 1000 mate 2 searchmoves e2e4 g1f3".parse::<Command>(),
            Ok(Command::Go(Go {
                depth: Some(7),
                nodes: Some(1000),
                mate: Some(2),
                move_time: Some(Duration::from_millis(100)),
                moves_to_search: vec!["e2e4".parse().unwrap(), "g1f3".parse().unwrap()],
                ..Default::default()
            },))
        );
        assert_eq!(
            "go wtime 1000 btime 1000 winc 100 binc 100 movestogo 5".parse::<Command>(),
            Ok(Command::Go(Go {
                time: ByColor {
                    white: Some(Duration::from_millis(1000)),
                    black: Some(Duration::from_millis(1000))
                },
                increment: ByColor {
                    white: Duration::from_millis(100),
                    black: Duration::from_millis(100),
                },
                moves_to_go: Some(5),
                ..Default::default()
            }))
        );
        assert_eq!(
            "go ponder".parse::<Command>(),
            Ok(Command::Go(Go {
                ponder: true,
                ..Default::default()
            }))
        );
    }
}
