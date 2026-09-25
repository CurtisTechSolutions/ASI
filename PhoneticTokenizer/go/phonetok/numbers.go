package phonetok

import (
	"regexp"
	"strconv"
	"strings"
)

// Numbers as words: what a digit sounds like is the word for it (the port of numbers.py).

var ones = []string{"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
	"twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"}
var tens = []string{"", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"}
var scales = []string{"", "thousand", "million", "billion", "trillion", "quadrillion", "quintillion", "sextillion",
	"septillion", "octillion", "nonillion", "decillion"}
var ordinals = map[string]string{"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
	"nine": "ninth", "twelve": "twelfth"}

// Cardinal is 1234 -> "one thousand two hundred thirty four"; negative numbers say "minus".
// The number is a decimal string so that it can be as long as the text makes it.
func Cardinal(digits string) string {
	digits = strings.TrimLeft(digits, "0")
	if digits == "" {
		return "zero"
	}
	// chunks of three from the right
	var chunks []string
	for len(digits) > 3 {
		chunks = append(chunks, digits[len(digits)-3:])
		digits = digits[:len(digits)-3]
	}
	chunks = append(chunks, digits)
	var parts []string
	for scale := len(chunks) - 1; scale >= 0; scale-- {
		n, _ := strconv.Atoi(chunks[scale])
		if n == 0 {
			continue
		}
		name := ""
		if scale < len(scales) {
			name = scales[scale]
		} else {
			name = "ten to the " + strconv.Itoa(scale*3)
		}
		part := smallCardinal(n)
		if name != "" {
			part += " " + name
		}
		parts = append(parts, part)
	}
	return strings.Join(parts, " ")
}

func smallCardinal(n int) string {
	if n < 20 {
		return ones[n]
	}
	if n < 100 {
		if n%10 == 0 {
			return tens[n/10]
		}
		return tens[n/10] + " " + ones[n%10]
	}
	out := ones[n/100] + " hundred"
	if n%100 != 0 {
		out += " " + smallCardinal(n%100)
	}
	return out
}

// Ordinal is 21 -> "twenty first", 100 -> "one hundredth".
func Ordinal(digits string) string {
	words := strings.Split(Cardinal(digits), " ")
	last := words[len(words)-1]
	switch {
	case ordinals[last] != "":
		words[len(words)-1] = ordinals[last]
	case strings.HasSuffix(last, "y"):
		words[len(words)-1] = last[:len(last)-1] + "ieth"
	default:
		words[len(words)-1] = last + "th"
	}
	return strings.Join(words, " ")
}

// Digits is each digit by name: "007" -> "zero zero seven".
func Digits(text string) string {
	var out []string
	for _, ch := range text {
		if ch >= '0' && ch <= '9' {
			out = append(out, ones[ch-'0'])
		}
	}
	return strings.Join(out, " ")
}

var numberRe = regexp.MustCompile(`^(?i)([$€£])?([-+−])?(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(st|nd|rd|th)?(%)?$`)

var currencies = map[string][2]string{"$": {"dollar", "dollars"}, "€": {"euro", "euros"}, "£": {"pound", "pounds"}}

// NumberWords is the words of a number token, or "" if it is not one.
func NumberWords(token string) string {
	m := numberRe.FindStringSubmatch(token)
	if m == nil {
		return ""
	}
	currency, sign, whole, frac, ordinal, percent := m[1], m[2], strings.ReplaceAll(m[3], ",", ""), m[4], m[5], m[6]
	isOne := strings.TrimLeft(whole, "0") == "1"
	var words []string
	if sign == "-" || sign == "−" {
		words = append(words, "minus")
	}
	hasFrac := m[4] != "" || strings.Contains(token, ".")
	switch {
	case ordinal != "":
		if hasFrac {
			return ""
		}
		words = append(words, Ordinal(whole))
	case currency != "" && hasFrac && len(frac) == 2:
		unit := currencies[currency]
		words = append(words, Cardinal(whole))
		if isOne {
			words = append(words, unit[0])
		} else {
			words = append(words, unit[1])
		}
		cents, _ := strconv.Atoi(frac)
		if cents != 0 {
			words = append(words, smallCardinal(cents))
			if cents == 1 {
				words = append(words, "cent")
			} else {
				words = append(words, "cents")
			}
		}
		return strings.Join(words, " ")
	default:
		words = append(words, Cardinal(whole))
		if hasFrac {
			words = append(words, "point", Digits(frac))
		}
	}
	if currency != "" {
		unit := currencies[currency]
		if isOne && !hasFrac {
			words = append(words, unit[0])
		} else {
			words = append(words, unit[1])
		}
	}
	if percent != "" {
		words = append(words, "percent")
	}
	return strings.Join(words, " ")
}

var runsRe = regexp.MustCompile(`\d+|[^\d]+`)

// SplitAlphanumeric is "mp3" -> ["mp", "3"]; a token without digits comes back alone.
func SplitAlphanumeric(token string) []string {
	if !strings.ContainsAny(token, "0123456789") {
		return []string{token}
	}
	return runsRe.FindAllString(token, -1)
}
