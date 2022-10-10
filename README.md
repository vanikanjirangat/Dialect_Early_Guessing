# Early guessing of dialects by incremental analysis

This is the implementation of our paper in EMNLP 2022 Findings: Early Guessing for Dialect Identification

The paper discuss about the experiments, which are primarily data-centric approaches for achieving the problem statement:
 *Detection of dialects before the full input utterance is given.*
We used model calibration approaches with BERT-based modelas and found generalizable criteria. (For more details, check the paper).

 
The experimnets are done on four datasets: Swiss German (GDI), Indo-Aryan (ILI), Arabic_Vardial (ADI), Arabic Online Commentary (AOC). The data can be found in the folder *data/*

The main script can be found in folder *scripts/*
