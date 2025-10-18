# Experiment: replicate HerdNet Results

Based on the paper "From crowd to herd counting: How to precisely detect and count African mammals using aerial imagery and deep learning?" by Alexandre Delplanque et al. 
we aim to replicate the results presented in the study using the HerdNet model for counting African mammals in aerial imagery.

It is also based on the original implementation available at [github](https://github.com/Alexandre-Delplanque/HerdNet)

## Structure

- `scripts`: Contains all the scripts used for data preprocessing, model training, and evaluation.

# Methodology presented in the paper

As a summary of the paper:

1. Data collection: The authors collected aerial images of African savannas available [online](https://dataverse.uliege.be/dataset.xhtml?persistentId=doi:10.58119/ULG/MIRUU5)
2. Data annotation: data was annotated using point annotations for each animal present in the images.
3. Patching: The images were divided into smaller patches to facilitate training, size of 512x512 pixels with an overlap of 160 pixels.
4. The model was trained using patches which only contained animals, excluding empty patches.
5. Then the model was over `train` full image to evaluate its performance and mine hard negatives.
6. Hard negative mining: Patches from areas where the model made incorrect predictions (false positives) were added to the training set to improve model robustness.
7. Retraining: The model was retrained using the augmented dataset that included hard negatives.