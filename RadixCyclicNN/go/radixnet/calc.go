package radixnet

// The calculator tool's expression language: numbers, operators and a handful
// of math functions, and nothing else.
//
// Python evaluates these with `ast` and a whitelist; Go has no `eval`, so this
// is a small recursive-descent parser over the same grammar.  It keeps
// Python's arithmetic where the two differ: `/` is always true division,
// `//` floors, `%` takes the sign of the divisor, int op int stays an int, and
// a result is rendered the way `str()` would render it.

import (
	"math"
	"sort"
	"strconv"
	"strings"
	"unicode"
)

// calcNames are the constants an expression may use.
var calcNames = map[string]float64{"pi": math.Pi, "e": math.E, "tau": 2 * math.Pi, "inf": math.Inf(1)}

// calcFunctions are the functions an expression may call, by the number of
// arguments each takes (-1: any).
var calcFunctions = map[string]int{
	"sqrt": 1, "floor": 1, "ceil": 1, "log": -1, "log2": 1, "log10": 1, "exp": 1, "sin": 1, "cos": 1,
	"tan": 1, "atan": 1, "hypot": -1, "fabs": 1, "pow": 2, "abs": 1, "round": -1, "min": -1, "max": -1,
	"sum": -1, "int": 1, "float": 1,
}

// number is a Python number: an int stays an int until something makes it a
// float, which is what decides how a result prints.
type number struct {
	i     int64
	f     float64
	isInt bool
	isBoo bool
}

func intValue(v int64) number     { return number{i: v, f: float64(v), isInt: true} }
func floatValue(v float64) number { return number{f: v} }
func boolValue(v bool) number {
	out := intValue(0)
	if v {
		out = intValue(1)
	}
	out.isBoo = true
	return out
}

// String renders the value as Python's str() would.
func (n number) String() string {
	switch {
	case n.isBoo:
		if n.i != 0 {
			return "True"
		}
		return "False"
	case n.isInt:
		return strconv.FormatInt(n.i, 10)
	}
	return pythonFloat(n.f)
}

// Float is the value as a float.
func (n number) Float() float64 {
	if n.isInt {
		return float64(n.i)
	}
	return n.f
}

func (n number) truthy() bool {
	if n.isInt {
		return n.i != 0
	}
	return n.f != 0
}

// pythonFloat renders a float the way Python's repr() does: the shortest form
// that round-trips, with a trailing ".0" on an integral value.
func pythonFloat(value float64) string {
	if math.IsInf(value, 1) {
		return "inf"
	}
	if math.IsInf(value, -1) {
		return "-inf"
	}
	if math.IsNaN(value) {
		return "nan"
	}
	text := strconv.FormatFloat(value, 'g', -1, 64)
	if e := strings.IndexAny(text, "eE"); e >= 0 { // Python writes 1e+16, Go writes 1e+16 too
		mantissa, exponent := text[:e], text[e+1:]
		if !strings.Contains(mantissa, ".") {
			mantissa += ".0"
		}
		sign := "+"
		if exponent[0] == '+' || exponent[0] == '-' {
			sign, exponent = string(exponent[0]), exponent[1:]
		}
		if len(exponent) < 2 {
			exponent = "0" + exponent
		}
		return mantissa + "e" + sign + exponent
	}
	if !strings.Contains(text, ".") {
		text += ".0"
	}
	return text
}

// SafeEval evaluates an arithmetic expression - numbers, operators and the
// math functions, nothing else.
func SafeEval(expression string) (string, error) {
	text := strings.TrimSpace(expression)
	if text == "" {
		return "", toolErrorf("the expression is empty")
	}
	if runeLen(text) > 500 {
		return "", toolErrorf("the expression is too long (500 characters max)")
	}
	parser := &calcParser{input: []rune(strings.ReplaceAll(text, "^", "**"))}
	value, err := parser.parseExpression()
	if err != nil {
		return "", err
	}
	parser.skipSpace()
	if parser.at < len(parser.input) {
		return "", toolErrorf("not an arithmetic expression: unexpected %s", pythonRepr(string(parser.input[parser.at])))
	}
	if len(value) != 1 {
		return "", toolErrorf("cannot evaluate the expression: it is a sequence, not a number")
	}
	return value[0].String(), nil
}

// calcParser is a recursive-descent parser over the expression grammar.  Every
// production returns a slice so a tuple or list can be passed to min / max /
// sum; a single value is a slice of one.
type calcParser struct {
	input []rune
	at    int
}

func (p *calcParser) skipSpace() {
	for p.at < len(p.input) && unicode.IsSpace(p.input[p.at]) {
		p.at++
	}
}

// accept consumes a literal token when it is next.
func (p *calcParser) accept(token string) bool {
	p.skipSpace()
	runes := []rune(token)
	if p.at+len(runes) > len(p.input) {
		return false
	}
	if string(p.input[p.at:p.at+len(runes)]) != token {
		return false
	}
	if isWordToken(token) { // a word must not be the prefix of a longer name
		after := p.at + len(runes)
		if after < len(p.input) && (unicode.IsLetter(p.input[after]) || unicode.IsDigit(p.input[after]) || p.input[after] == '_') {
			return false
		}
	}
	p.at += len(runes)
	return true
}

func isWordToken(token string) bool {
	for _, r := range token {
		if !unicode.IsLetter(r) {
			return false
		}
	}
	return token != ""
}

// peek reports whether a literal token is next without consuming it.
func (p *calcParser) peek(token string) bool {
	mark := p.at
	if p.accept(token) {
		p.at = mark
		return true
	}
	return false
}

// parseExpression is the top of the grammar: `or`.
func (p *calcParser) parseExpression() ([]number, error) {
	left, err := p.parseAnd()
	if err != nil {
		return nil, err
	}
	for p.accept("or") {
		right, err := p.parseAnd()
		if err != nil {
			return nil, err
		}
		one, err := single(left)
		if err != nil {
			return nil, err
		}
		if one.truthy() { // Python's `or` returns the operand, not a bool
			continue
		}
		left = right
	}
	return left, nil
}

func (p *calcParser) parseAnd() ([]number, error) {
	left, err := p.parseNot()
	if err != nil {
		return nil, err
	}
	for p.accept("and") {
		right, err := p.parseNot()
		if err != nil {
			return nil, err
		}
		one, err := single(left)
		if err != nil {
			return nil, err
		}
		if !one.truthy() {
			continue
		}
		left = right
	}
	return left, nil
}

func (p *calcParser) parseNot() ([]number, error) {
	if p.accept("not") {
		value, err := p.parseNot()
		if err != nil {
			return nil, err
		}
		one, err := single(value)
		if err != nil {
			return nil, err
		}
		return []number{boolValue(!one.truthy())}, nil
	}
	return p.parseComparison()
}

var comparisons = []string{"==", "!=", "<=", ">=", "<", ">"}

func (p *calcParser) parseComparison() ([]number, error) {
	left, err := p.parseSum()
	if err != nil {
		return nil, err
	}
	for {
		matched := ""
		for _, op := range comparisons {
			if p.accept(op) {
				matched = op
				break
			}
		}
		if matched == "" {
			return left, nil
		}
		right, err := p.parseSum()
		if err != nil {
			return nil, err
		}
		a, err := single(left)
		if err != nil {
			return nil, err
		}
		b, err := single(right)
		if err != nil {
			return nil, err
		}
		out := false
		switch matched {
		case "==":
			out = a.Float() == b.Float()
		case "!=":
			out = a.Float() != b.Float()
		case "<":
			out = a.Float() < b.Float()
		case "<=":
			out = a.Float() <= b.Float()
		case ">":
			out = a.Float() > b.Float()
		case ">=":
			out = a.Float() >= b.Float()
		}
		left = []number{boolValue(out)}
	}
}

func (p *calcParser) parseSum() ([]number, error) {
	left, err := p.parseProduct()
	if err != nil {
		return nil, err
	}
	for {
		var op string
		switch {
		case p.peek("+"):
			p.accept("+")
			op = "+"
		case p.peek("-"):
			p.accept("-")
			op = "-"
		default:
			return left, nil
		}
		right, err := p.parseProduct()
		if err != nil {
			return nil, err
		}
		value, err := arithmetic(op, left, right)
		if err != nil {
			return nil, err
		}
		left = []number{value}
	}
}

func (p *calcParser) parseProduct() ([]number, error) {
	left, err := p.parseUnary()
	if err != nil {
		return nil, err
	}
	for {
		var op string
		switch {
		case p.peek("//"):
			p.accept("//")
			op = "//"
		case p.peek("*") && !p.peek("**"):
			p.accept("*")
			op = "*"
		case p.peek("**"):
			return left, nil // handled by parseUnary's power
		case p.peek("/"):
			p.accept("/")
			op = "/"
		case p.peek("%"):
			p.accept("%")
			op = "%"
		default:
			return left, nil
		}
		right, err := p.parseUnary()
		if err != nil {
			return nil, err
		}
		value, err := arithmetic(op, left, right)
		if err != nil {
			return nil, err
		}
		left = []number{value}
	}
}

func (p *calcParser) parseUnary() ([]number, error) {
	if p.accept("-") {
		value, err := p.parseUnary()
		if err != nil {
			return nil, err
		}
		one, err := single(value)
		if err != nil {
			return nil, err
		}
		if one.isInt {
			return []number{intValue(-one.i)}, nil
		}
		return []number{floatValue(-one.f)}, nil
	}
	if p.accept("+") {
		return p.parseUnary()
	}
	return p.parsePower()
}

func (p *calcParser) parsePower() ([]number, error) {
	base, err := p.parseAtom()
	if err != nil {
		return nil, err
	}
	if p.accept("**") {
		exponent, err := p.parseUnary() // right-associative, and -2 ** -1 is legal
		if err != nil {
			return nil, err
		}
		value, err := arithmetic("**", base, exponent)
		if err != nil {
			return nil, err
		}
		return []number{value}, nil
	}
	return base, nil
}

func (p *calcParser) parseAtom() ([]number, error) {
	p.skipSpace()
	if p.at >= len(p.input) {
		return nil, toolErrorf("not an arithmetic expression: it ends too early")
	}
	switch r := p.input[p.at]; {
	case r == '(' || r == '[':
		closing := ")"
		if r == '[' {
			closing = "]"
		}
		p.at++
		values, err := p.parseSequence(closing)
		if err != nil {
			return nil, err
		}
		return values, nil
	case unicode.IsDigit(r) || r == '.':
		return p.parseNumber()
	case unicode.IsLetter(r) || r == '_':
		return p.parseName()
	case r == '\'' || r == '"':
		return nil, toolErrorf("only numbers are allowed in an expression")
	}
	return nil, toolErrorf("not an arithmetic expression: unexpected %s", pythonRepr(string(p.input[p.at])))
}

// parseSequence reads the comma-separated values up to a closing bracket.
func (p *calcParser) parseSequence(closing string) ([]number, error) {
	values := []number{}
	if p.accept(closing) {
		return values, nil
	}
	for {
		value, err := p.parseExpression()
		if err != nil {
			return nil, err
		}
		values = append(values, value...)
		if p.accept(",") {
			if p.accept(closing) { // a trailing comma
				return values, nil
			}
			continue
		}
		if p.accept(closing) {
			return values, nil
		}
		return nil, toolErrorf("not an arithmetic expression: %s is missing", pythonRepr(closing))
	}
}

func (p *calcParser) parseNumber() ([]number, error) {
	start := p.at
	seenDot, seenExp := false, false
	for p.at < len(p.input) {
		r := p.input[p.at]
		switch {
		case unicode.IsDigit(r), r == '_':
			p.at++
		case r == '.' && !seenDot && !seenExp:
			seenDot = true
			p.at++
		case (r == 'e' || r == 'E') && !seenExp && p.at+1 < len(p.input) &&
			(unicode.IsDigit(p.input[p.at+1]) || p.input[p.at+1] == '+' || p.input[p.at+1] == '-'):
			seenExp = true
			p.at += 2
		default:
			goto done
		}
	}
done:
	text := strings.ReplaceAll(string(p.input[start:p.at]), "_", "")
	if !seenDot && !seenExp {
		value, err := strconv.ParseInt(text, 10, 64)
		if err == nil {
			return []number{intValue(value)}, nil
		}
	}
	value, err := strconv.ParseFloat(text, 64)
	if err != nil {
		return nil, toolErrorf("not an arithmetic expression: %s is not a number", pythonRepr(text))
	}
	return []number{floatValue(value)}, nil
}

func (p *calcParser) parseName() ([]number, error) {
	start := p.at
	for p.at < len(p.input) {
		r := p.input[p.at]
		if !unicode.IsLetter(r) && !unicode.IsDigit(r) && r != '_' {
			break
		}
		p.at++
	}
	name := string(p.input[start:p.at])
	if p.accept("(") {
		args, err := p.parseSequence(")")
		if err != nil {
			return nil, err
		}
		if _, ok := calcFunctions[name]; !ok {
			return nil, toolErrorf("unknown function %s (have: %s)", pythonRepr(name), strings.Join(calcFunctionNames(), ", "))
		}
		value, err := callCalcFunction(name, args)
		if err != nil {
			return nil, err
		}
		return []number{value}, nil
	}
	if value, ok := calcNames[name]; ok {
		return []number{floatValue(value)}, nil
	}
	if _, ok := calcFunctions[name]; ok {
		return nil, toolErrorf("unknown name %s", pythonRepr(name)) // a function is not a value
	}
	return nil, toolErrorf("unknown name %s", pythonRepr(name))
}

func calcFunctionNames() []string {
	names := make([]string, 0, len(calcFunctions))
	for name := range calcFunctions {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// single is the one value of a production, or an error when it is a sequence.
func single(values []number) (number, error) {
	if len(values) != 1 {
		return number{}, toolErrorf("cannot evaluate the expression: a sequence where a number was wanted")
	}
	return values[0], nil
}

// arithmetic applies one binary operator, keeping Python's int / float rules.
func arithmetic(op string, left, right []number) (number, error) {
	a, err := single(left)
	if err != nil {
		return number{}, err
	}
	b, err := single(right)
	if err != nil {
		return number{}, err
	}
	bothInt := a.isInt && b.isInt
	switch op {
	case "+":
		if bothInt {
			return intValue(a.i + b.i), nil
		}
		return floatValue(a.Float() + b.Float()), nil
	case "-":
		if bothInt {
			return intValue(a.i - b.i), nil
		}
		return floatValue(a.Float() - b.Float()), nil
	case "*":
		if bothInt {
			return intValue(a.i * b.i), nil
		}
		return floatValue(a.Float() * b.Float()), nil
	case "/":
		if b.Float() == 0 {
			return number{}, toolErrorf("division by zero")
		}
		return floatValue(a.Float() / b.Float()), nil
	case "//":
		if b.Float() == 0 {
			return number{}, toolErrorf("division by zero")
		}
		if bothInt {
			return intValue(int64(math.Floor(float64(a.i) / float64(b.i)))), nil
		}
		return floatValue(math.Floor(a.Float() / b.Float())), nil
	case "%":
		if b.Float() == 0 {
			return number{}, toolErrorf("division by zero")
		}
		if bothInt {
			out := a.i % b.i // Python's modulo takes the sign of the divisor
			if out != 0 && (out < 0) != (b.i < 0) {
				out += b.i
			}
			return intValue(out), nil
		}
		out := math.Mod(a.Float(), b.Float())
		if out != 0 && (out < 0) != (b.Float() < 0) {
			out += b.Float()
		}
		return floatValue(out), nil
	case "**":
		if bothInt && b.i >= 0 {
			return intValue(int64(math.Pow(float64(a.i), float64(b.i)))), nil
		}
		return floatValue(math.Pow(a.Float(), b.Float())), nil
	}
	return number{}, toolErrorf("cannot evaluate the expression: unknown operator %s", pythonRepr(op))
}

// callCalcFunction applies one of the whitelisted functions.
func callCalcFunction(name string, args []number) (number, error) {
	wanted := calcFunctions[name]
	if wanted >= 0 && len(args) != wanted {
		return number{}, toolErrorf("cannot evaluate the expression: %s() takes %d argument(s), got %d", name, wanted, len(args))
	}
	if len(args) == 0 && name != "sum" {
		return number{}, toolErrorf("cannot evaluate the expression: %s() needs an argument", name)
	}
	one := number{}
	if len(args) > 0 {
		one = args[0]
	}
	switch name {
	case "sqrt":
		if one.Float() < 0 {
			return number{}, toolErrorf("cannot evaluate the expression: math domain error")
		}
		return floatValue(math.Sqrt(one.Float())), nil
	case "floor":
		return intValue(int64(math.Floor(one.Float()))), nil
	case "ceil":
		return intValue(int64(math.Ceil(one.Float()))), nil
	case "log":
		if len(args) == 2 {
			if one.Float() <= 0 || args[1].Float() <= 0 {
				return number{}, toolErrorf("cannot evaluate the expression: math domain error")
			}
			return floatValue(math.Log(one.Float()) / math.Log(args[1].Float())), nil
		}
		if len(args) != 1 {
			return number{}, toolErrorf("cannot evaluate the expression: log() takes 1 or 2 arguments")
		}
		if one.Float() <= 0 {
			return number{}, toolErrorf("cannot evaluate the expression: math domain error")
		}
		return floatValue(math.Log(one.Float())), nil
	case "log2", "log10":
		if one.Float() <= 0 {
			return number{}, toolErrorf("cannot evaluate the expression: math domain error")
		}
		if name == "log2" {
			return floatValue(math.Log2(one.Float())), nil
		}
		return floatValue(math.Log10(one.Float())), nil
	case "exp":
		return floatValue(math.Exp(one.Float())), nil
	case "sin":
		return floatValue(math.Sin(one.Float())), nil
	case "cos":
		return floatValue(math.Cos(one.Float())), nil
	case "tan":
		return floatValue(math.Tan(one.Float())), nil
	case "atan":
		return floatValue(math.Atan(one.Float())), nil
	case "hypot":
		total := 0.0
		for _, arg := range args {
			total += arg.Float() * arg.Float()
		}
		return floatValue(math.Sqrt(total)), nil
	case "fabs":
		return floatValue(math.Abs(one.Float())), nil
	case "pow":
		return floatValue(math.Pow(one.Float(), args[1].Float())), nil
	case "abs":
		if one.isInt {
			if one.i < 0 {
				return intValue(-one.i), nil
			}
			return intValue(one.i), nil
		}
		return floatValue(math.Abs(one.f)), nil
	case "round":
		if len(args) == 2 {
			scale := math.Pow(10, args[1].Float())
			return floatValue(math.RoundToEven(one.Float()*scale) / scale), nil
		}
		if len(args) != 1 {
			return number{}, toolErrorf("cannot evaluate the expression: round() takes 1 or 2 arguments")
		}
		if one.isInt {
			return one, nil
		}
		return intValue(int64(math.RoundToEven(one.Float()))), nil // Python rounds half to even
	case "min", "max":
		best := args[0]
		for _, arg := range args[1:] {
			if (name == "min") == (arg.Float() < best.Float()) && arg.Float() != best.Float() {
				best = arg
			}
		}
		return best, nil
	case "sum":
		total := intValue(0)
		for _, arg := range args {
			value, err := arithmetic("+", []number{total}, []number{arg})
			if err != nil {
				return number{}, err
			}
			total = value
		}
		return total, nil
	case "int":
		return intValue(int64(one.Float())), nil
	case "float":
		return floatValue(one.Float()), nil
	}
	return number{}, toolErrorf("unknown function %s (have: %s)", pythonRepr(name), strings.Join(calcFunctionNames(), ", "))
}
