all:
	cd rascil-base-root; make build;make tag_latest;make push_latest
	cd rascil-base;make build;make tag_latest;make push_latest
	cd rascil-full-root; make build;make tag_latest;make push_latest
	cd rascil-full; make build;make tag_latest;make push_latest
	cd rascil-notebook; make build;make tag_latest;make push_latest

rm_all:
	cd rascil-base-root; make rm
	cd rascil-base;make rm
	cd rascil-full-root; make rm
	cd rascil-full; make rm
	cd rascil-notebook; make rm
