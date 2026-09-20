//! The one timestamp the model file carries.
//!
//! `saved_at` is written as Python writes it -
//! `datetime.now(timezone.utc).isoformat(timespec="seconds")`, e.g.
//! `2026-09-20T01:23:45+00:00` - so the three implementations' files look the
//! same to anything that reads the field.  The standard library has no calendar,
//! so this is the civil-from-days algorithm, which is twenty lines.

use std::time::{SystemTime, UNIX_EPOCH};

/// An ISO-8601 UTC timestamp to the second.
pub fn utc_now() -> String {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    iso8601(seconds)
}

/// An ISO-8601 UTC timestamp for a Unix time in seconds.
pub fn iso8601(unix_seconds: i64) -> String {
    let days = unix_seconds.div_euclid(86_400);
    let secs = unix_seconds.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}+00:00",
        secs / 3600,
        (secs / 60) % 60,
        secs % 60
    )
}

/// Howard Hinnant's `civil_from_days`: a day count since 1970-01-01 as a date.
fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097); // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_reads_like_pythons_isoformat() {
        // datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="seconds")
        assert_eq!(iso8601(0), "1970-01-01T00:00:00+00:00");
        assert_eq!(iso8601(1_000_000_000), "2001-09-09T01:46:40+00:00");
        assert_eq!(iso8601(1_788_000_000), "2026-08-29T10:40:00+00:00");
        assert_eq!(iso8601(951_782_400), "2000-02-29T00:00:00+00:00"); // a leap day
        assert_eq!(utc_now().len(), "1970-01-01T00:00:00+00:00".len());
    }
}
