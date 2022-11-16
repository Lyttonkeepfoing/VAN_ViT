# VAN_ViT
## Datasets
This section is dedicated to the datasets used in the paper: download and formatting instructions are provided 
for experiment replication purposes.

### IAM

#### Details

IAM corresponds to english grayscale handwriting images (from the LOB corpus).
We provide a script to format this dataset for the commonly used split for result comparison purposes.
The different splits are as follow:

|           | train | validation |  test |
|:---------:|:-----:|:----------:|:-----:|
|    line   | 6,482 |     976    | 2,915 |
| paragraph |  747  |     116    |  336  |

#### Download



- Register at the [FKI's webpage](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database)
- Download the dataset [here](https://fki.tic.heia-fr.ch/databases/download-the-iam-handwriting-database) 
- Move the following files into the folder Datasets/raw/IAM/
    - formsA-D.tgz
    - formsE-H.tgz
    - formsI-Z.tgz
    - lines.tgz
    - ascii.tgz



### RIMES

#### Details

RIMES corresponds to french grayscale handwriting images.
We provide a script to format this dataset for the commonly used split for result comparison purposes.
The different splits are as follow:

|           | train | validation |  test |
|:---------:|:-----:|:----------:|:-----:|
|    line   | 9,947 |     1,333  | 778 |
| paragraph |  1400 |     100    |  100 |

#### Download

- Fill in the a2ia user agreement form available [here](http://www.a2ialab.com/doku.php?id=rimes_database:start) and send it by email to rimesnda@a2ia.com. You will receive by mail a username and a password
- Login in and download the data from [here](http://www.a2ialab.com/doku.php?id=rimes_database:data:icdar2011:line:icdar2011competitionline)
- Move the following files into the folder Datasets/raw/RIMES/
    - eval_2011_annotated.xml
    - eval_2011_gray.tar
    - training_2011_gray.tar
    - training_2011.xml

### READ 2016

#### Details
READ 2016 corresponds to Early Modern German RGB handwriting images.
We provide a script to format this dataset for the commonly used split for result comparison purposes.
The different splits are as follow:

|           | train | validation |  test |
|:---------:|:-----:|:----------:|:-----:|
|    line   | 8,349 |  1,040    | 1,138|
| paragraph |  1584 |     179    | 197 |

#### Download

- From root folder:

```
cd Datasets/raw
mkdir READ_2016
cd READ_2016
wget https://zenodo.org/record/1164045/files/{Test-ICFHR-2016.tgz,Train-And-Val-ICFHR-2016.tgz}
```


### Format the datasets

- Comment/Uncomment the following lines from the main function of the script "format_datasets.py" according to your needs and run it

```
if __name__ == "__main__":

    # format_IAM_line()
    # format_IAM_paragraph()

    # format_RIMES_line()
    # format_RIMES_paragraph()

    # format_READ2016_line()
    # format_READ2016_paragraph()
```

- This will generate well-formated datasets, usable by the training scripts.
