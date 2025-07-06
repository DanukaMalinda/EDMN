import sys
import os
import numpy as np

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.yeast import yeast
from data.wine import wine
from data.video_game_sales import vgame
from data.turk_student_eval import turk
from data.theorem import thrm
from data.bike import bike
from data.blog_feedback import blog
from data.concrete import conc
from data.contraceptive import contra
from data.diamonds import diam
from data.drugs import drugs
from data.energy import ener
from data.fifa19 import fifa
from data.rice import rice
from data.pendigits import pend
from data.optdigits import optd
from data.nursery import nurse
from data.news_popularity import news
from data.microbes import micro
from data.human_activity_recognition import hars
from data.gasdrift import gasd
from data.gesture import gest
from data.adult import adult
from data.avila import avila
from data.bike_binary import bike_b
from data.blog_feedback_binary import blog_b
from data.breast_cancer_cont import bc_cont
from data.cars_binary import cars_b
from data.concrete_binary import conc_b
from data.contraceptive_binary import contra_b
from data.credit_appl import cappl
from data.diamonds_binary import diam_b
from data.dota2 import dota_b
from data.drugs_binary import drugs_b
from data.energy_binary import ener_b
from data.fifa19_binary import fifa_b
from data.flare import flare_b
from data.grid_stability import grid_b
from data.internet_ads import ads
from data.magic import magic
from data.mini_boone import boone
from data.mushroom import mush
from data.yeast_binary import yeast_b
from data.wine_binary import wine_b
from data.video_game_sales_binary import vgame_b
from data.turk_student_eval_binary import turk_b
from data.theorem_binary import thrm_b
from data.superconductor_binary import cond_b
from data.student_performance import study_b
from data.spambase import spam_b
from data.skillcraft_binary import craft_b
from data.mini_boone import boone
from data.student_alcohol import alco_b
from data.phishing import phish
from data.news_popularity_binary import news_b
from data.musk import musk
from data.music import music
from data.occupancy import occup


def load_data(dataset):
    if dataset == 'yeast':
        X_n, y_n = yeast.load_data()

    elif dataset == 'boone':
        X_n, y_n = boone.load_data()

    elif dataset == 'music':
        X_n, y_n = music.load_data()

    elif dataset == 'wine':
        X_n, y_n = wine.load_data()

    elif dataset == 'occup':
        X_n, y_n = occup.load_data()

    elif dataset == 'vgame':
        X_n, y_n = vgame.load_data()

    elif dataset == 'turk':
        X_n, y_n = turk.load_data()

    elif dataset == 'thrm':
        X_n, y_n = thrm.load_data()

    elif dataset == 'bike':
        X_n, y_n = bike.load_data()

    elif dataset == 'blog':
        X_n, y_n = blog.load_data()

    elif dataset == 'conc':
        X_n, y_n = conc.load_data()

    elif dataset == 'contra':
        X_n, y_n = contra.load_data()

    elif dataset == 'diam':
        X_n, y_n = diam.load_data()
    
    elif dataset == 'drugs':
        X_n, y_n = drugs.load_data()

    elif dataset == 'ener':
        X_n, y_n = ener.load_data()

    elif dataset == 'fifa':
        X_n, y_n = fifa.load_data()

    elif dataset == 'gasd':
        X_n, y_n = gasd.load_data()

    elif dataset == 'gest':
        X_n, y_n = gest.load_data()

    elif dataset == 'hars':
        X_n, y_n = hars.load_data()

    elif dataset == 'insec':
        X_n, y_n = wine.load_data()

    elif dataset == 'micro':
        X_n, y_n = micro.load_data()

    elif dataset == 'news':
        X_n, y_n = news.load_data()

    elif dataset == 'news_b':
        X_n, y_n = news_b.load_data()

    elif dataset == 'nurse':
        X_n, y_n = nurse.load_data()

    elif dataset == 'optd':
        X_n, y_n = optd.load_data()

    elif dataset == 'pend':
        X_n, y_n = pend.load_data()

    elif dataset == 'rice':
        X_n, y_n = rice.load_data()

    elif dataset == 'adult':
        X_n, y_n = adult.load_data()

    elif dataset == 'avila':
        X_n, y_n = avila.load_data()

    elif dataset == 'bike_b':
        X_n, y_n = bike_b.load_data()

    elif dataset == 'blog_b':
        X_n, y_n = blog_b.load_data()

    elif dataset == 'bc_cont':
        X_n, y_n = bc_cont.load_data()

    elif dataset == 'cars_b':
        X_n, y_n = cars_b.load_data()

    elif dataset == 'conc_b':
        X_n, y_n = conc_b.load_data()

    elif dataset == 'contra_b':
        X_n, y_n = contra_b.load_data()

    elif dataset == 'cappl':
        X_n, y_n = cappl.load_data()

    elif dataset == 'diam_b':
        X_n, y_n = diam_b.load_data()

    elif dataset == 'dota_b':
        X_n, y_n = dota_b.load_data()

    elif dataset == 'drugs_b':
        X_n, y_n = drugs_b.load_data()

    elif dataset == 'ener_b':
        X_n, y_n = ener_b.load_data()

    elif dataset == 'fifa_b':
        X_n, y_n = fifa_b.load_data()

    elif dataset == 'flare_b':
        X_n, y_n = flare_b.load_data()

    elif dataset == 'grid_b':
        X_n, y_n = grid_b.load_data()

    elif dataset == 'ads':
        X_n, y_n = ads.load_data()

    elif dataset == 'magic':
        X_n, y_n = magic.load_data()

    elif dataset == 'boone':
        X_n, y_n = boone.load_data()

    elif dataset == 'mush':
        X_n, y_n = mush.load_data()

    elif dataset == 'yeast_b':
        X_n, y_n = yeast_b.load_data()

    elif dataset == 'wine_b':
        X_n, y_n = wine_b.load_data()

    elif dataset == 'vgame_b':
        X_n, y_n = vgame_b.load_data()

    elif dataset == 'turk_b':
        X_n, y_n = turk_b.load_data()

    elif dataset == 'thrm_b':
        X_n, y_n = thrm_b.load_data()

    elif dataset == 'cond_b':
        X_n, y_n = cond_b.load_data()

    elif dataset == 'study_b':
        X_n, y_n = study_b.load_data()

    elif dataset == 'spam_b':
        X_n, y_n = spam_b.load_data()
        
    elif dataset == 'craft_b':
        X_n, y_n = craft_b.load_data()

    elif dataset == 'alco':
        X_n, y_n = alco_b.load_data()

    elif dataset == 'phish':
        X_n, y_n = phish.load_data()

    elif dataset == 'musk':
        X_n, y_n = musk.load_data()

    else:
        raise ValueError("Unknown dataset: " + dataset)

    return X_n, y_n

# Example usage
def main():
    X, y = load_data('occup')
    print(X.shape, y.shape)

    print(np.unique(y, return_counts=True))

if __name__ == '__main__':
    main()