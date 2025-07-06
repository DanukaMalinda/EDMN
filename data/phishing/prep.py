import pandas as pd

"""
data preparation.
Reference:
    - https://github.com/z-donyavi/MC-SQ/blob/main/helpers.py
    Z. Donyavi, A. B. S. Serapi˜ao, and G. Batista, “Mc-sq and mc-mq: Ensembles for multiclass quantification,
    ” IEEE Transactions on Knowledge and Data Engineering, vol. 36, no. 8, pp. 4007–4019, 2024.

    - https://github.com/tobiasschumacher/quantification_paper/blob/master/helpers.py
    T. Schumacher, M. Strohmaier, and F. Lemmerich, 
    “A comparative evaluation of quantification methods,” arXiv preprint arXiv:2103.03223, 2021.

"""

def prep_data(binned=False):
    f = "../data/phishing/dataset.csv"

    # colnames = ['index',
    #             'having_IP_Address',
    #             'URL_Length',
    #             'Shortining_Service',
    #             'having_At_Symbol',
    #             'double_slash_redirecting',
    #             'Prefix_Suffix',
    #             'having_Sub_Domain',
    #             'SSLfinal_State',
    #             'Domain_registeration_length',
    #             'Favicon',
    #             'port',
    #             'HTTPS_token',
    #             'Request_URL',
    #             'URL_of_Anchor',
    #             'Links_in_tags',
    #             'SFH',
    #             'Submitting_to_email',
    #             'Abnormal_URL',
    #             'Redirect',
    #             'on_mouseover',
    #             'RightClick',
    #             'popUpWidnow',
    #             'Iframe',
    #             'age_of_domain',
    #             'DNSRecord',
    #             'web_traffic',
    #             'Page_Rank',
    #             'Google_Index',
    #             'Links_pointing_to_page',
    #             'Statistical_report',
    #             'Result']

    dta = pd.read_csv(f,
                        # names=colnames,
                      index_col=False,
                      skipinitialspace=True)
    
    
    
    dta['Result'] = dta['Result'].replace({-1: 0, 1: 1}).astype(int)

    return dta

# dta.to_pickle("dta.pkl")
