#!/usr/bin/octave

function Z = sigmoid(X)
	 Z = 1 ./ (1 .+ (e.^-X));
endfunction
