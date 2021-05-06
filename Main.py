"""
This file is the main file that connect all the 3 other files to a single program.
"""

# Imports
import ctypes
import random
import pygame.display

import Data_Processor as dp
import GUI_Handler as gh
import Logistic_Regression_AI as lr


def start_guess(screen, model, dic):
    """
    Function that let the user enter news titles for the AI to guess.
    Taking new titles until the program is closed or the home button is clicked.
    """
    gh.set_screen(screen, 4)
    title = gh.wait_text_input(screen, 655, 790, 35, 4, "")
    guessing = True
    if type(title) is bool:
        guessing = False
    while guessing:
        predict, output = lr.guess_one_title(title, model, dic)
        if predict:
            gh.set_screen(screen, 6)
            title = gh.wait_text_input(screen, 693, 810, 35, 6, output)
        else:
            gh.set_screen(screen, 5)
            title = gh.wait_text_input(screen, 693, 810, 35, 5, output)
        if type(title) is bool:
            guessing = False


def game(test_df, model, dic, screen):
    """
    Function that start the "Guessing game".
    The function shows a random title from the training data (so the AI didn't "saw" it before).
    The user guess if it's Fake or not, then the AI does the same.
    The function tracks the points of the user and AI, and shows suitable output on the screen.
    """
    points = [0, 0]
    gh.set_screen(screen, 8)
    title = ""
    while len(title.split()) < 5 or len(title.split()) > 17:
        game_title = random.randint(0, len(test_df))
        title = test_df.iloc[game_title][0]
    gh.show_text_on_box(screen, title, 35, 40, 495)
    answer = gh.wait_yes_no(760, True)
    playing = True
    if type(answer) is not bool:
        playing = False
    else:
        answer = int(answer)
    while playing:
        true_answer = int(test_df.iloc[game_title][1])
        if answer == true_answer:
            points[1] += 1
            if true_answer == 1:
                gh.set_screen(screen, 10)
            else:
                gh.set_screen(screen, 10)
        else:
            if true_answer == 1:
                gh.set_screen(screen, 9)
            else:
                gh.set_screen(screen, 9)
        ai_guess = lr.guess_one_title(title, model, dic)
        log = ai_guess[1] + ". "
        if ai_guess[0]:
            if true_answer == 0:
                log += "He is wrong..."
            else:
                log += "He is right!"
                points[0] += 1
        else:
            if true_answer == 1:
                log += "He is wrong..."
            else:
                log += "He is right!"
                points[0] += 1
        title = ""
        while len(title.split()) < 5 or len(title.split()) > 17:
            game_title = random.randint(0, len(test_df))
            title = test_df.iloc[game_title][0]
        log += " ||The AI: " + str(points[0]) + " points, you: " + str(points[1]) + " points||"
        gh.show_text_on_box(screen, log, 35, 40, 555)
        gh.show_text_on_box(screen, title, 35, 40, 680)
        pygame.display.flip()
        answer = gh.wait_yes_no(910, True)
        if type(answer) is not bool:
            playing = False
        else:
            answer = int(answer)


def start_program():
    """
    The function that connects everything to a single program.
    Starts by letting the user to choose if he wants the AI to relearn the data.
    Then if he wants to see the confusion matrix.
    And then he can choose between giving new titles to the AI or playing the guessing game.
    As well he can move freely between this option, using the home button.
    """
    df = dp.load_data()
    screen = gh.init_pygame()
    gh.set_screen(screen, 0)

    relearn = gh.wait_yes_no(760, False)
    if relearn:
        sure = ctypes.windll.user32.MessageBoxW(0, "The relearning proccess will take a while and will use a lot of proccessing power that may slow your computer\nAre you sure you want the AI to relearn the data?", "Are you sure?", 4)
        if sure == 6:
            gh.set_screen(screen, 3)
            train_size = 0.8
            X_train, Y_train, X_test, Y_test, dic, test_df = dp.train_test_vectorization(df, "title", train_size)
            model = lr.learn(X_train, Y_train, X_test, Y_test, dic, test_df)
            acc, cf = lr.test(model, X_test, Y_test)
            gh.set_screen(screen, 2)
            print("----Relearn process completed:", acc*100, "Percent success rate----\n\n")
        else:
            relearn = False
    if not relearn:
        model, X_test, Y_test, dic, test_df = lr.load_model()
        acc, cf = lr.test(model, X_test, Y_test)
        gh.set_screen(screen, 1)
        print("----Model loaded.", acc*100, "Percent Success rate----\n\n")

    present_cf = gh.wait_yes_no(760, False)
    if present_cf:
        gh.show_cf(cf)

    while True:
        gh.set_screen(screen, 11)
        action = gh.wait_action()
        if action == 1:
            start_guess(screen, model, dic)
        else:
            game(test_df, model, dic, screen)


start_program()
