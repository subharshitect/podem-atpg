# Makefile (Unipolar SPWM + V/f control)
PY ?= python3
OUTDIR ?= out
PNG ?= $(OUTDIR)/pwm.png

# Frequencies + sampling
FREF  ?= 50.0
FCAR  ?= 2000.0
FS    ?= 200000
PHASE ?= 0.0

# Carrier and DC bus
CRR   ?= 1.0
VDC   ?= 1.0

# V/f settings (enable with VF=1)
VF     ?= 1
FBASE  ?= 50.0
ABASE  ?= 0.9
AMAX   ?= 0.98
VBOOST ?= 0.0
FBOOST ?= 5.0

# If VF=0, this A is used directly
A ?= 0.9

.PHONY: plot show clean

plot:
	@mkdir -p $(OUTDIR)
	$(PY) unipolar_spwm.py \
		--crr $(CRR) --fref $(FREF) --fcar $(FCAR) --fs $(FS) --phase $(PHASE) --vdc $(VDC) \
		--vf $(VF) --fbase $(FBASE) --abase $(ABASE) --amax $(AMAX) --vboost $(VBOOST) --fboost $(FBOOST) \
		--A $(A) \
		--out $(PNG)
	@echo "Wrote: $(PNG)"

show: plot
	@echo "Opening: $(PNG)"
	code $(PNG)

clean:
	@rm -rf $(OUTDIR)

# make show VF=1 FREF=2 VBOOST=0.15 FBOOST=5
# make show VF=1 FREF=10
