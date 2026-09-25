module github.com/CurtisTechSolutions/ASI/RadixCyclicNN/go

go 1.24

// The phonetic tokenizer lives beside this project in the same repository: the
// phone and syllable units of the encoding read text through it.  Still no
// third-party dependency.
require github.com/CurtisTechSolutions/ASI/PhoneticTokenizer/go v0.0.0

replace github.com/CurtisTechSolutions/ASI/PhoneticTokenizer/go => ../../PhoneticTokenizer/go
