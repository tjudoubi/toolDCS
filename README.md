# toolDCS
The data in "DCS_CAT_data_2022" represents the sample of experimental results conducted by CAT on the latest data crawled from DCS.

The data in "DCS_CIT_data_2022" represents the sample of experimental results conducted by CIT on the latest data crawled from DCS.

The data in "DCS_CAT_data_2021" represents the sample of experimental results conducted by CAT on the data from SIT(ICSE2019) and CIT(ASE2021).

The data in "DCS_CIT_data_2021" represents the sample of experimental results conducted by CIT on the data from SIT(ICSE2019) and CIT(ASE2021).

In "NEWData":
* "2021" is the experimental results conducted by DCS on the data from SIT(ICSE2019) and CIT(ASE2021).
* "2022" is the experimental results conducted by DCS on the latest data crawled from DCS.


Code Explanation:


1. Get the translations:

    * In toolDCS, we first need to get translations of the from translation software. 

2. Then we use "convert_NEWData_txt.py" convert source sentence and translations to token sequence.

3. Get the mapping relationship between Chinese tokens and English tokens:

    * We use awesome-align(https://github.com/neulab/awesome-align) to get corresponding mapping relationship in "alignment".
    * Then, in this step, we use "align-awesome.py" to obtain a dictionary of word alignment mappings between an English source sentence and three Chinese sentences from three translation softwares.
    
4. By using SLAHAN, we can obtain the main part of the source sentence:

    * We use SLAHAN(https://github.com/kamigaito/SLAHAN) to get main part of source sentence.

5. Then we use "get_origin_parserTree.py" to obtain main part and adjunct parts for each sentences and corresponding mapping chinese for each parts. 

6. Finally we use "experiment_first.py" to make Differential testing based on Compositional Semantics.