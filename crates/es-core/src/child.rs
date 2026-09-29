//! Starting a Python child the Windows loader may fail to start (packet M15/R1).
//!
//! Many interpreters importing `torch` (CUDA) at once on Windows sometimes fail to start: an
//! `OSError: [WinError 6] ... Error loading "...\c10.dll"`, or an NTSTATUS exit such as
//! `0xC000070A` with nothing written. The cause is the DLL loader under concurrent starts, not
//! the code, so a start that failed that way is tried again, a bounded number of times, and
//! each retry is announced on stderr so a flaky start is never silent.
//!
//! Only a *start* is retried -- the spawn and the first line the child owes -- never a child
//! that dies once it is running, which would change results. A retried start is a fresh
//! process with the same program, arguments and environment; no seed or state carries over
//! from the failed one, so it changes no result byte.

use std::fmt::Display;
use std::time::Duration;

/// How many times a start is tried.
pub const START_ATTEMPTS: u32 = 3;

/// The pause before the second and before the third try.
const BACKOFF: [Duration; (START_ATTEMPTS - 1) as usize] =
    [Duration::from_millis(500), Duration::from_millis(1500)];

/// Whether a child that failed while starting failed the way the Windows loader fails
/// concurrent starts: `code` is its exit code (`None`: still running, or a Unix signal),
/// `stderr` what it wrote there, `printed` whether it wrote any of the line it owed on stdout.
///
/// Transient when the exit code is an NTSTATUS error (`0xC000_0000..=0xCFFF_FFFF`, negative
/// as an `i32`: the OS ended the process), when stderr names `[WinError 6]` or a DLL that did
/// not load (`Error loading "...dll"`), or when the child ended having written nothing at
/// all. Anything else -- a clean `ImportError`, a script error, an error line on stdout -- is
/// a real answer and is not retried. A missing interpreter never gets here: the spawn fails.
pub fn transient_start_failure(code: Option<i32>, stderr: &str, printed: bool) -> bool {
    let ntstatus = code.is_some_and(|c| (c as u32) >> 28 == 0xC);
    let loader = stderr.contains("[WinError 6]")
        || (stderr.contains("Error loading \"") && stderr.to_ascii_lowercase().contains(".dll\""));
    let silent = !printed && stderr.trim().is_empty();
    ntstatus || loader || silent
}

/// Runs `start` until it succeeds, fails with a failure it marks as not transient (see
/// [`transient_start_failure`]), or has run [`START_ATTEMPTS`] times, pausing 0.5 s and then
/// 1.5 s in between. Each retry prints one line on stderr naming `what`, the attempt and the
/// failure. `Err` is the last failure and how many attempts were made, for the caller to add
/// to its message when there was more than one.
///
/// `start` must start a fresh child each time with the same program, arguments and
/// environment, so a retried start produces the same bytes a first-time one would.
pub fn retry_start<T, E: Display>(
    what: &str,
    mut start: impl FnMut() -> Result<T, (E, bool)>,
) -> Result<T, (E, u32)> {
    let mut attempt = 1;
    loop {
        match start() {
            Ok(value) => return Ok(value),
            Err((e, true)) if attempt < START_ATTEMPTS => {
                eprintln!(
                    "{what}: start attempt {attempt} of {START_ATTEMPTS} failed ({}); retrying",
                    gist(&e.to_string())
                );
                std::thread::sleep(BACKOFF[attempt as usize - 1]);
                attempt += 1;
            }
            Err((e, _)) => return Err((e, attempt)),
        }
    }
}

/// A failure message on one line: its first line, and its last where there are more (a
/// traceback's last line is the exception), cut at 300 characters.
fn gist(text: &str) -> String {
    let mut lines = text.lines().map(str::trim).filter(|l| !l.is_empty());
    let first = lines.next().unwrap_or("no message");
    let line = match lines.next_back() {
        Some(last) => format!("{first} ... {last}"),
        None => first.to_owned(),
    };
    match line.char_indices().nth(300) {
        Some((cut, _)) => format!("{}...", &line[..cut]),
        None => line,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const WIN_ERROR_6: &str = "Traceback (most recent call last):\n  File \"<string>\", line 1\n\
        OSError: [WinError 6] The handle is invalid. Error loading \
        \"C:\\venv\\Lib\\site-packages\\torch\\lib\\c10.dll\" or one of its dependencies.\n";

    /// An NTSTATUS as `ExitStatus::code` reports it on Windows.
    fn nt(status: u32) -> i32 {
        i32::from_ne_bytes(status.to_ne_bytes())
    }

    #[test]
    fn an_ntstatus_exit_is_transient() {
        // 0xC000070A with nothing written, as the CLI tests saw it.
        assert!(transient_start_failure(Some(nt(0xC000_070A)), "", false));
        // An access violation after the child printed its line is still the OS ending it.
        assert!(transient_start_failure(Some(nt(0xC000_0005)), "", true));
        // -1 (`sys.exit(-1)`) is negative but no NTSTATUS error.
        assert!(!transient_start_failure(Some(-1), "SystemExit", false));
    }

    #[test]
    fn a_dll_the_loader_could_not_load_is_transient() {
        assert!(transient_start_failure(Some(1), WIN_ERROR_6, false));
        let other = "OSError: [WinError 1455] The paging file is too small. Error loading \
                     \"C:\\torch\\lib\\cudnn64_9.DLL\" or one of its dependencies.";
        assert!(transient_start_failure(Some(1), other, false));
    }

    #[test]
    fn a_child_that_ended_saying_nothing_is_transient() {
        assert!(transient_start_failure(Some(1), "  \n", false));
        assert!(transient_start_failure(None, "", false));
    }

    #[test]
    fn a_clean_import_error_is_not_transient() {
        let stderr = "ModuleNotFoundError: No module named 'torch'\n";
        assert!(!transient_start_failure(Some(1), stderr, false));
        // The reference scripts report it as a protocol line on stdout and nothing on stderr.
        assert!(!transient_start_failure(Some(1), "", true));
    }

    #[test]
    fn a_missing_interpreter_is_not_transient() {
        // What a shell or `cmd` says for one; a direct spawn fails before any exit code.
        assert!(!transient_start_failure(
            Some(127),
            "sh: python: command not found",
            false
        ));
        assert!(!transient_start_failure(
            Some(9009),
            "'python' is not recognized as an internal or external command",
            false
        ));
    }

    #[test]
    fn transient_failures_are_retried_until_the_start_succeeds() {
        let mut calls = 0;
        let got = retry_start("stand-in", || {
            calls += 1;
            if calls < 3 {
                Err((format!("crashed\n{WIN_ERROR_6}"), true))
            } else {
                Ok(calls)
            }
        });
        assert_eq!(got.unwrap(), 3);
    }

    #[test]
    fn a_failure_that_is_not_transient_is_not_retried() {
        let mut calls = 0;
        let got: Result<(), _> = retry_start("stand-in", || {
            calls += 1;
            Err(("ModuleNotFoundError: No module named 'torch'", false))
        });
        assert_eq!(got.unwrap_err().1, 1);
        assert_eq!(calls, 1);
    }

    #[test]
    fn the_retries_are_bounded() {
        let mut calls = 0;
        let got: Result<(), _> = retry_start("stand-in", || {
            calls += 1;
            Err(("exit status 0xC000070A", true))
        });
        assert_eq!(got.unwrap_err(), ("exit status 0xC000070A", START_ATTEMPTS));
        assert_eq!(calls, START_ATTEMPTS);
    }

    #[test]
    fn the_gist_is_one_line() {
        assert_eq!(
            gist("python cannot import (exit status 1):\nTraceback\n  x\nOSError: boom\n"),
            "python cannot import (exit status 1): ... OSError: boom"
        );
        assert_eq!(gist(""), "no message");
        assert_eq!(gist(&"x".repeat(400)).len(), 303);
    }
}
