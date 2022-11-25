FILM := $(shell seq 0 2)
NUMBERS := $(shell seq 0 8)
LIM := 10
R := 500

# non-jpegs
FILMS := $(addprefix film,${FILM})

JOBS0 := $(addprefix job0,${NUMBERS})
JOBS1 := $(addprefix job1,${NUMBERS})
JOBS2 := $(addprefix job2,${NUMBERS})

${JOBS0}: job0%: ; @python3 codec.py -f 0 -v stats -r $(R) -c $(LIM) -q $*
${JOBS1}: job1%: ; @python3 codec.py -f 1 -v stats -r $(R) -c $(LIM) -q $*
${JOBS2}: job2%: ; @python3 codec.py -f 2 -v stats -r $(R) -c $(LIM) -q $*

film0: ${JOBS0}
film1: ${JOBS1}
film2: ${JOBS2}

all-films: ${FILMS}

# jpegs
FILMSJ := $(addprefix filmj,${FILM})

JOBS0J := $(addprefix jobj0,${NUMBERS})
JOBS1J := $(addprefix jobj1,${NUMBERS})
JOBS2J := $(addprefix jobj2,${NUMBERS})

${JOBS0J}: jobj0%: ; @python3 codec.py -f 0 -v stats -c $(LIM) -j -q $*
${JOBS1J}: jobj1%: ; @python3 codec.py -f 1 -v stats -c $(LIM) -j -q $*
${JOBS2J}: jobj2%: ; @python3 codec.py -f 2 -v stats -c $(LIM) -j -q $*

filmj0: ${JOBS0J}
filmj1: ${JOBS1J}
filmj2: ${JOBS2J}

all-films-jpeg: ${FILMSJ}

all: all-films all-films-jpeg
