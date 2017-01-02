# SMLP
Semi-supervised label propagation (SMLP) is a project by Yuan Liu (yuanl4), Yangyi Lu (yangyil), Ben Parr (bparr).
We will consider a new method for semi-supervised handwriting recognition by using unlabeled data to learn more information in addition to labeled data. There are some existing ways to solve this problem, like using Euclidean distance to determine if the data is close. In this project, we will use a neural network computed inner-product to do semi-supervised clustering instead of Euclidean distance. This semi-supervised learning could expand learning to the vast amounts of unlabeled documents.

Final Report (12/7/16): https://www.overleaf.com/read/sbsmcczgmnmt

Midterm Report (11/9/16): https://www.overleaf.com/read/dsjtcfjndhch

Proposal (10/2/16): https://www.overleaf.com/read/vmwvhyvrrmdx

Code use:

Outputs are saved in the saver/ directory.
words_SMLP_MC_train.py constains the words list, as well as the width and height of image.
words_SMLP_output.py outputs saver/{train, test}.txt, where the each line is the label (index of word in word list).
label_propagation.py reads saver/{train, test}.txt and outputs accurracy of the label propagation.
