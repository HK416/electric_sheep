//! What `lerobot-train` prints, read while it runs: the training bar and the metric lines.

use std::io::{Read, Write};
use std::process::{Command, Stdio};

// --- `lerobot-train`'s console (packet M12/R2) ------------------------------------------

/// What one `\r`- or `\n`-delimited piece of `lerobot-train`'s console says. tqdm's bar and
/// the logger both write stderr, so a metric line can land on the end of the bar it
/// interrupted, in the same piece.
#[derive(Debug, Default, PartialEq)]
pub(super) struct LerobotSaid {
    /// `Training:  70%|███████   | 3510/5000 [03:51<01:39, 14.94step/s]`: the exact step, the
    /// total, and steps per second once tqdm has measured one.
    pub(super) tqdm: Option<(u64, u64, Option<f64>)>,
    /// `… step:3K smpl:26K ep:49 epch:0.25 loss:0.131 grdn:12.233 lr:1.0e-04 …`, every
    /// `log_freq` steps. Its `step:` is abbreviated and is not read.
    pub(super) loss: Option<f64>,
    pub(super) lr: Option<f64>,
}

pub(super) fn lerobot_said(piece: &str) -> LerobotSaid {
    let token = |key: &str| {
        piece
            .split_whitespace()
            .find_map(|t| t.strip_prefix(key)?.parse().ok())
    };
    LerobotSaid {
        tqdm: training_bar(piece),
        loss: token("loss:"),
        lr: token("lr:"),
    }
}

/// The training bar only: the pretrained backbone's download is a tqdm bar too.
pub(super) fn training_bar(piece: &str) -> Option<(u64, u64, Option<f64>)> {
    let bar = piece.split_once("Training:")?.1;
    let (counts, timing) = bar.splitn(3, '|').nth(2)?.trim_start().split_once(" [")?;
    let (step, total) = counts.split_once('/')?;
    let rate = timing.split_once(", ")?.1.split(']').next()?.trim();
    let rate = match (rate.strip_suffix("step/s"), rate.strip_suffix("s/step")) {
        (Some(per_s), _) => per_s.parse().ok(),
        (_, Some(s_per)) => s_per.parse().ok().map(|s: f64| 1.0 / s),
        _ => None,
    };
    Some((step.parse().ok()?, total.parse().ok()?, rate))
}

/// The latest bar, and the stream-5 rows the metric lines complete.
#[derive(Default)]
pub(super) struct LerobotProgress {
    step: Option<u64>,
    rate: Option<f64>,
}

impl LerobotProgress {
    /// `[step, loss, lr, samples_per_s]` when this piece gives a loss and a bar has already
    /// given the exact step; nothing otherwise, never a stand-in for the loss.
    pub(super) fn read(&mut self, piece: &str, batch: Option<u32>) -> Option<[f64; 4]> {
        let said = lerobot_said(piece);
        if let Some((step, _, rate)) = said.tqdm {
            (self.step, self.rate) = (Some(step), rate);
        }
        Some([
            self.step? as f64,
            said.loss?,
            said.lr.unwrap_or(f64::NAN),
            self.rate
                .zip(batch)
                .map_or(f64::NAN, |(r, b)| r * f64::from(b)),
        ])
    }
}

/// Copies one of the trainer's pipes to ours as it arrives, byte for byte, and hands each
/// `\r`- or `\n`-delimited piece on. It keeps draining after a failed write, so the trainer
/// never blocks on a full pipe.
pub(super) fn relay(mut from: impl Read, mut to: impl Write, mut on_piece: impl FnMut(String)) {
    let (mut buf, mut piece) = ([0u8; 4096], Vec::new());
    loop {
        let n = match from.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => break,
        };
        let _ = to.write_all(&buf[..n]).and_then(|()| to.flush());
        for &b in &buf[..n] {
            if b != b'\r' && b != b'\n' {
                piece.push(b);
            } else if !piece.is_empty() {
                on_piece(String::from_utf8_lossy(&piece).into_owned());
                piece.clear();
            }
        }
    }
    if !piece.is_empty() {
        on_piece(String::from_utf8_lossy(&piece).into_owned());
    }
}

/// `lerobot-train` with someone listening: both pipes relayed to ours, and every piece of what
/// it said handed to `on_piece`. Nothing it prints is written anywhere new.
pub(super) fn stream_lerobot(
    cmd: &mut Command,
    mut on_piece: impl FnMut(&str),
) -> std::io::Result<std::process::ExitStatus> {
    let mut child = cmd.stdout(Stdio::piped()).stderr(Stdio::piped()).spawn()?;
    let (tx, rx) = std::sync::mpsc::channel();
    if let Some(out) = child.stdout.take() {
        let tx = tx.clone();
        std::thread::spawn(move || relay(out, std::io::stdout(), |p| drop(tx.send(p))));
    }
    if let Some(err) = child.stderr.take() {
        let tx = tx.clone();
        std::thread::spawn(move || relay(err, std::io::stderr(), |p| drop(tx.send(p))));
    }
    drop(tx);
    // Ends when both relays have seen their pipe close.
    for piece in rx {
        on_piece(&piece);
    }
    child.wait()
}
