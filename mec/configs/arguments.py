from .default_config import *
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-train', '--train', action='store_true',
    help='train model')
parser.add_argument('-test', '--test', action='store_true',
    help='evaluate model on test set')
parser.add_argument('-c', '--continue_training', action='store_true',
    help='continue training from last point')
parser.add_argument('-score', '--score', action='store_true',
    help='calc precision, recall and F1, then write to an excel file')
parser.add_argument('-prod', '--prod', action='store_true',
    help='test production per image')
parser.add_argument('-mix', '--mix', action='store_true',
    help='output image mix matrix as xlsx file')
parser.add_argument('-d', '--deploy', action='store_true',
    help='generate index to wiki_idx json file')
parser.add_argument('-lr', '--learning_rate', type=float,
    help='designating statis training rate')
parser.add_argument('-e', '--epochs', type=int,
    help='how many epochs to train in this run')
parser.add_argument('-p', '--path', type=str,
    help='path to store results')
parser.add_argument('-manager', '--start_manager', action='store_true',
    help='train model')
parser.add_argument('-workers', '--start_workers', action='store_true',
    help='train model')

args = parser.parse_args()

if args.train:
    train=True
if args.test:
    test=True
if args.score:
    score=True
if args.prod:
    prod=True
if args.mix:
    mix=True
if args.deploy:
    deploy=True
if args.continue_training:
    continue_training=True
if args.learning_rate:
    learning_rate = args.learning_rate
if args.epochs:
    epochs = args.epochs
if args.path:
    path = args.path
    history_filename        = os.path.join(path,history_filename)
    model_filename          = os.path.join(path,model_filename)
    best_model_filename     = os.path.join(path,best_model_filename)