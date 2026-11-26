package httpapi

import (
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"path/filepath"

	"github.com/gin-gonic/gin"
)

func documentProxy(worker *url.URL) gin.HandlerFunc {
	proxy := &httputil.ReverseProxy{Rewrite: func(req *httputil.ProxyRequest) {
		req.SetURL(worker)
		req.Out.Header.Del("Cookie")
		req.Out.Header.Del("Authorization")
	}}
	return gin.WrapH(proxy)
}

func staticFiles(dir string) gin.HandlerFunc {
	files := http.FileServer(http.Dir(dir))
	return func(c *gin.Context) {
		r := c.Request
		if r.Method != http.MethodGet && r.Method != http.MethodHead {
			c.Header("Allow", "GET, HEAD")
			c.Status(http.StatusMethodNotAllowed)
			c.Writer.WriteHeaderNow()
			return
		}
		info, err := os.Stat(filepath.Join(dir, filepath.FromSlash(r.URL.Path)))
		if err != nil || (info.IsDir() && r.URL.Path != "/") {
			http.NotFound(c.Writer, r)
			return
		}
		c.Status(http.StatusOK)
		files.ServeHTTP(c.Writer, r)
	}
}
