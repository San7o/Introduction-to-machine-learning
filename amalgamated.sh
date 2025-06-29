#!/bin/sh

OUT="ML-amalgamated.md"
TEX_OUT="Intro-to-Machine-Learning.tex"

cat 02\ -\ Basics.md > $OUT
cat 03\ -\ KNN.md >> $OUT
cat 04\ -\ Linear\ Models.md >> $OUT
cat 05\ -\ Beyond\ Binary\ Classification.md >> $OUT
cat 06\ -\ Decision\ Tree.md >> $OUT
cat 07\ -\ Gradient\ Descent.md >> $OUT
cat 08\ -\ Support\ Vector\ Machines.md >> $OUT
cat 08.2\ -\ Non\ linearly-separable\ data.md >> $OUT
cat 09\ -\ Regularization.md >> $OUT
cat 10\ -\ Unsupervised\ Learning.md >> $OUT
cat 11\ -\ Clustering.md >> $OUT
cat 12\ -\ Intro\ to\ NN.md >> $OUT
cat 13\ -\ Deep\ Generative\ Models.md >> $OUT
cat 14\ -\ Diffusion\ models.md >> $OUT
cat 15\ -\ Reinforcement\ Learning.md >> $OUT

pandoc $OUT -t latex -o $TEX_OUT.tmp
cat tex-start.txt > $TEX_OUT
cat $TEX_OUT.tmp >> $TEX_OUT
rm $TEX_OUT.tmp
cat tex-end.txt >> $TEX_OUT
texi2pdf $TEX_OUT
