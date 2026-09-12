package server

import (
	"bytes"
	"io"
	"mime"
	"mime/multipart"
)

// parseMultipart returns the file parts of a multipart/form-data body.
func parseMultipart(body []byte, boundary string) ([]uploadPayload, error) {
	mr := multipart.NewReader(bytes.NewReader(body), boundary)
	var files []uploadPayload
	for {
		part, err := mr.NextPart()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		filename := part.FileName()
		if filename == "" {
			// plain form fields are ignored
			_, _ = io.Copy(io.Discard, part)
			continue
		}
		data, err := io.ReadAll(part)
		if err != nil {
			return nil, err
		}
		if _, params, perr := mime.ParseMediaType(part.Header.Get("Content-Disposition")); perr == nil && params["filename"] != "" {
			filename = params["filename"]
		}
		files = append(files, uploadPayload{name: filename, data: data})
	}
	return files, nil
}
