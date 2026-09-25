//! Numbers as words (the port of numbers.py).

const ONES: [&str; 20] = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
];
const TENS: [&str; 10] = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
];
const SCALES: [&str; 12] = [
    "",
    "thousand",
    "million",
    "billion",
    "trillion",
    "quadrillion",
    "quintillion",
    "sextillion",
    "septillion",
    "octillion",
    "nonillion",
    "decillion",
];

fn small_cardinal(n: usize) -> String {
    if n < 20 {
        return ONES[n].to_string();
    }
    if n < 100 {
        return if n % 10 == 0 {
            TENS[n / 10].to_string()
        } else {
            format!("{} {}", TENS[n / 10], ONES[n % 10])
        };
    }
    let mut out = format!("{} hundred", ONES[n / 100]);
    if n % 100 != 0 {
        out.push(' ');
        out.push_str(&small_cardinal(n % 100));
    }
    out
}

/// `"1234"` -> "one thousand two hundred thirty four" (a decimal string, any length).
pub fn cardinal(digits: &str) -> String {
    let digits = digits.trim_start_matches('0');
    if digits.is_empty() {
        return "zero".to_string();
    }
    let bytes = digits.as_bytes();
    let mut chunks: Vec<&str> = Vec::new();
    let mut end = bytes.len();
    while end > 3 {
        chunks.push(&digits[end - 3..end]);
        end -= 3;
    }
    chunks.push(&digits[..end]);
    let mut parts: Vec<String> = Vec::new();
    for scale in (0..chunks.len()).rev() {
        let n: usize = chunks[scale].parse().unwrap_or(0);
        if n == 0 {
            continue;
        }
        let name = if scale < SCALES.len() {
            SCALES[scale].to_string()
        } else {
            format!("ten to the {}", scale * 3)
        };
        let mut part = small_cardinal(n);
        if !name.is_empty() {
            part.push(' ');
            part.push_str(&name);
        }
        parts.push(part);
    }
    parts.join(" ")
}

/// `"21"` -> "twenty first".
pub fn ordinal(digits: &str) -> String {
    let mut words: Vec<String> = cardinal(digits).split(' ').map(|s| s.to_string()).collect();
    let last = words.last().cloned().unwrap_or_default();
    let replaced = match last.as_str() {
        "one" => "first".to_string(),
        "two" => "second".to_string(),
        "three" => "third".to_string(),
        "five" => "fifth".to_string(),
        "eight" => "eighth".to_string(),
        "nine" => "ninth".to_string(),
        "twelve" => "twelfth".to_string(),
        w if w.ends_with('y') => format!("{}ieth", &w[..w.len() - 1]),
        w => format!("{w}th"),
    };
    *words.last_mut().unwrap() = replaced;
    words.join(" ")
}

/// Each digit by name.
pub fn digits(text: &str) -> String {
    text.chars()
        .filter(|c| c.is_ascii_digit())
        .map(|c| ONES[(c as u8 - b'0') as usize])
        .collect::<Vec<_>>()
        .join(" ")
}

fn currency_words(symbol: char) -> (&'static str, &'static str) {
    match symbol {
        '$' => ("dollar", "dollars"),
        '€' => ("euro", "euros"),
        _ => ("pound", "pounds"),
    }
}

/// The words of a number token, or `None` if it is not one.
pub fn number_words(token: &str) -> Option<String> {
    let chars: Vec<char> = token.chars().collect();
    let mut i = 0;
    let mut currency: Option<char> = None;
    if i < chars.len() && matches!(chars[i], '$' | '€' | '£') {
        currency = Some(chars[i]);
        i += 1;
    }
    let mut sign = None;
    if i < chars.len() && matches!(chars[i], '-' | '+' | '−') {
        sign = Some(chars[i]);
        i += 1;
    }
    // the integer part: \d{1,3}(,\d{3})+ or \d+
    let start = i;
    while i < chars.len() && chars[i].is_ascii_digit() {
        i += 1;
    }
    if i == start {
        return None;
    }
    let mut whole: String = chars[start..i].iter().collect();
    if i < chars.len() && chars[i] == ',' && (i - start) <= 3 {
        // comma groups, all or nothing
        let mut j = i;
        let mut grouped = whole.clone();
        let mut ok = true;
        while j < chars.len() && chars[j] == ',' {
            if chars[j + 1..].len() >= 3 && chars[j + 1..j + 4].iter().all(|c| c.is_ascii_digit()) {
                grouped.extend(chars[j + 1..j + 4].iter());
                j += 4;
            } else {
                ok = false;
                break;
            }
        }
        if ok && j > i {
            whole = grouped;
            i = j;
        }
    }
    let mut frac: Option<String> = None;
    if i < chars.len() && chars[i] == '.' {
        let fstart = i + 1;
        let mut j = fstart;
        while j < chars.len() && chars[j].is_ascii_digit() {
            j += 1;
        }
        if j == fstart {
            return None;
        }
        frac = Some(chars[fstart..j].iter().collect());
        i = j;
    }
    let mut ordinal_suffix = false;
    if i + 2 <= chars.len() {
        let two: String = chars[i..i + 2].iter().collect::<String>().to_lowercase();
        if matches!(two.as_str(), "st" | "nd" | "rd" | "th") {
            ordinal_suffix = true;
            i += 2;
        }
    }
    let mut percent = false;
    if i < chars.len() && chars[i] == '%' {
        percent = true;
        i += 1;
    }
    if i != chars.len() {
        return None;
    }
    let is_one = whole.trim_start_matches('0') == "1";
    let mut words: Vec<String> = Vec::new();
    if matches!(sign, Some('-') | Some('−')) {
        words.push("minus".to_string());
    }
    if ordinal_suffix {
        if frac.is_some() {
            return None;
        }
        words.push(ordinal(&whole));
    } else if let (Some(c), Some(f)) = (currency, frac.as_ref().filter(|f| f.len() == 2)) {
        let unit = currency_words(c);
        words.push(cardinal(&whole));
        words.push((if is_one { unit.0 } else { unit.1 }).to_string());
        let cents: usize = f.parse().unwrap_or(0);
        if cents != 0 {
            words.push(small_cardinal(cents));
            words.push((if cents == 1 { "cent" } else { "cents" }).to_string());
        }
        return Some(words.join(" "));
    } else {
        words.push(cardinal(&whole));
        if let Some(f) = &frac {
            words.push("point".to_string());
            words.push(digits(f));
        }
    }
    if let Some(c) = currency {
        let unit = currency_words(c);
        words.push(
            (if is_one && frac.is_none() {
                unit.0
            } else {
                unit.1
            })
            .to_string(),
        );
    }
    if percent {
        words.push("percent".to_string());
    }
    Some(words.join(" "))
}

/// `"mp3"` -> ["mp", "3"]; a token without digits comes back alone.
pub fn split_alphanumeric(token: &str) -> Vec<String> {
    if !token.chars().any(|c| c.is_ascii_digit()) {
        return vec![token.to_string()];
    }
    let mut out: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut in_digits: Option<bool> = None;
    for c in token.chars() {
        let d = c.is_ascii_digit();
        if in_digits.is_some() && in_digits != Some(d) {
            out.push(std::mem::take(&mut current));
        }
        in_digits = Some(d);
        current.push(c);
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}
