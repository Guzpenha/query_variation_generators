import xml.etree.ElementTree as ET
from IPython import embed
import pandas as pd
import ir_datasets


def main():
    dataset_names = [
        'trec-robust04/fold1',
        'trec-robust04/fold2',
        'trec-robust04/fold3',
        'trec-robust04/fold4',
        'trec-robust04/fold5',
        'gov2/trec-tb-2004',
        'gov2/trec-tb-2005',
        'gov2/trec-tb-2006',
        'aquaint/trec-robust-2005',
        'gov/trec-web-2002',
        'gov/trec-web-2003',
        'clueweb12/b13/ntcir-www-2',
        'clueweb12/b13/ntcir-www-3',
        'clueweb12/b13/trec-misinfo-2019',
        'cord19/trec-covid',
        'cord19/trec-covid/round1',
        'cord19/trec-covid/round2',
        'cord19/trec-covid/round3',
        'cord19/trec-covid/round4',
        'cord19/trec-covid/round5'        
    ]
    def clean_str(s):
        return s.replace("\n", " ").replace("\"", "")

    final_dataset = []
    
    for dataset_name in dataset_names:
        dataset = ir_datasets.load(dataset_name)
        for t in dataset.queries_iter():
            final_dataset.append([clean_str(t.description), clean_str(t.title), dataset_name])
    
    for dd_trec in ['dd_trec_2015', 'dd_trec_2016', 'dd_trec_2017']:        
        root = ET.parse('{}.xml'.format(dd_trec)).getroot()
        for topic in root.findall('domain/topic'):
            topic_text = topic.get('name')
            description = topic[0].text
            final_dataset.append([clean_str(description), clean_str(topic_text), dd_trec])        

    pd.DataFrame(final_dataset, columns=['description', 'title', 'dataset']).to_csv("trec_desc_to_title.csv", index=False)

if __name__ == "__main__":
    main()