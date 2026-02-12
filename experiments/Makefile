PY ?= python3
OUTDIR ?= out
RPTDIR ?= $(OUTDIR)/report

# Default command frequency
FREF  ?= 15.0
MR    ?= 20
FCAR  ?= $(shell echo "$(FREF) * $(MR)" | bc)

FS    ?= 200000
PHASE ?= 0.0
CRR   ?= 1.0
VDC   ?= 1.0

VF     ?= 1
FBASE  ?= 50.0
ABASE  ?= 0.9
AMAX   ?= 0.98
VBOOST ?= 0.0
FBOOST ?= 5.0
A      ?= 0.9

QBITS ?= 0

LOAD ?= 1
RVAL ?= 1.0
LVAL ?= 0.01

FREQS ?= 2 5 10 15 25 50

RUN_PNG  ?= $(RPTDIR)/run_waveforms.png
RUN_NPZ  ?= $(RPTDIR)/run_dump.npz

.PHONY: run spectrum sweep report show clean

run:
	@mkdir -p $(RPTDIR)
	$(PY) unipolar_spwm.py \
		--crr $(CRR) --fref $(FREF) --fcar $(FCAR) --fs $(FS) --phase $(PHASE) --vdc $(VDC) \
		--vf $(VF) --fbase $(FBASE) --abase $(ABASE) --amax $(AMAX) --vboost $(VBOOST) --fboost $(FBOOST) \
		--A $(A) --qbits $(QBITS) \
		--load $(LOAD) --r $(RVAL) --l $(LVAL) \
		--out $(RUN_PNG) --dump $(RUN_NPZ) --tag "single run"
	@echo "Wrote: $(RUN_PNG)"
	@echo "Wrote: $(RUN_NPZ)"

spectrum: run
	@mkdir -p $(RPTDIR)/analysis
	$(PY) report_analysis.py --inp $(RUN_NPZ) --outdir $(RPTDIR)/analysis --spec_max_hz $(shell echo "$(FCAR) * 10" | bc)
	@echo "Wrote spectrum + metrics under: $(RPTDIR)/analysis"

sweep:
	@mkdir -p $(RPTDIR)
	$(PY) sweep_report.py \
		--py $(PY) --outdir $(RPTDIR)/sweep \
		--freqs "$(FREQS)" --mr $(MR) \
		--fs $(FS) --phase $(PHASE) --crr $(CRR) --vdc $(VDC) \
		--vf $(VF) --fbase $(FBASE) --abase $(ABASE) --amax $(AMAX) --vboost $(VBOOST) --fboost $(FBOOST) --A $(A) \
		--qbits $(QBITS) --load $(LOAD) --r $(RVAL) --l $(LVAL)
	@echo "Wrote sweep artifacts under: $(RPTDIR)/sweep"

show: report
	code $(RPTDIR)

clean:
	@rm -rf $(OUTDIR)

FOURIER_K ?= 25
DEADTIME_SAMPLES ?= 20

fourier: run
	$(PY) fourier_series_plot.py --inp $(RPTDIR)/run_dump.npz --K $(FOURIER_K) --outdir $(RPTDIR)/analysis

deadtime: run
	$(PY) deadtime_plot.py --inp $(RPTDIR)/run_dump.npz --dt $(DEADTIME_SAMPLES) --out $(RPTDIR)/analysis/deadtime_timing.png --start 0 --span 2000

report: spectrum fourier deadtime sweep
	@echo "All report plots generated under: $(RPTDIR)"
