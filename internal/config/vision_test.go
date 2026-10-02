package config

import (
	"errors"
	"testing"
)

func TestResolveVision(t *testing.T) {
	noEnv := func(string) string { return "" }
	cases := []struct {
		name       string
		flags      Flags
		env        map[string]string
		profile    string
		wantErr    bool
		wantView   bool
		wantText   bool
		wantImages int
	}{
		{name: "images from the flag", flags: Flags{Images: []string{"a.png", "b.png"}}, wantImages: 2},
		{name: "view_image off by default"},
		{name: "view_image from the profile entry", profile: `{"local":{"viewImage":true}}`, wantView: true},
		{name: "view_image from env", env: map[string]string{EnvViewImage: "1"}, wantView: true},
		{name: "flag beats env", flags: Flags{ViewImage: bp(false)}, env: map[string]string{EnvViewImage: "1"}},
		{name: "text-only model without images is fine", profile: `{"local":{"textOnly":true}}`, wantText: true},
		{name: "text-only model refuses --image", flags: Flags{Images: []string{"a.png"}}, profile: `{"local":{"textOnly":true}}`, wantErr: true},
		{name: "text-only model refuses view_image", profile: `{"local":{"textOnly":true,"viewImage":true}}`, wantErr: true},
		{name: "--text-only flag refuses --image", flags: Flags{TextOnly: bp(true), Images: []string{"a.png"}}, wantErr: true},
		{name: "--text-only=false clears a profile mark", flags: Flags{TextOnly: bp(false), Images: []string{"a.png"}}, profile: `{"local":{"textOnly":true}}`, wantImages: 1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			getenv := noEnv
			if tc.env != nil {
				getenv = func(k string) string { return tc.env[k] }
			}
			path := ""
			if tc.profile != "" {
				path = writeProfile(t, tc.profile)
			}
			cfg, err := Resolve(tc.flags, getenv, path)
			if tc.wantErr {
				if !errors.Is(err, ErrTextOnlyModel) {
					t.Fatalf("err = %v, want ErrTextOnlyModel", err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if cfg.ViewImage != tc.wantView || cfg.TextOnly != tc.wantText || len(cfg.Images) != tc.wantImages {
				t.Errorf("got view=%t textOnly=%t images=%d, want %t/%t/%d", cfg.ViewImage, cfg.TextOnly, len(cfg.Images), tc.wantView, tc.wantText, tc.wantImages)
			}
		})
	}
}
