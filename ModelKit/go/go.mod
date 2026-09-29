module github.com/CurtisTechSolutions/ASI/ModelKit/go

go 1.24

// What is built on the count / reward model rather than the model itself - the
// kit, the HTTP API and the radixnet-count CLI - over the model and the
// phonetic tokenizer, which live beside this module in the same repository.
// Still no third-party dependency.
require (
	github.com/CurtisTechSolutions/ASI/PhoneticTokenizer/go v0.0.0
	github.com/CurtisTechSolutions/ASI/RadixCyclicNN/go v0.0.0
)

replace (
	github.com/CurtisTechSolutions/ASI/PhoneticTokenizer/go => ../../PhoneticTokenizer/go
	github.com/CurtisTechSolutions/ASI/RadixCyclicNN/go => ../../RadixCyclicNN/go
)
