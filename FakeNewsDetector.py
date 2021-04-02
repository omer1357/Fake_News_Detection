import ctypes

import Data_Processor as dp
import GUI_Handler as gh
import Logistic_Regression_AI as lr


def start_program():
    df = dp.load_data()
    screen = gh.init_pygame()
    gh.set_screen(screen, 0)

    relearn = gh.wait_yes_no()
    if relearn:
        sure = ctypes.windll.user32.MessageBoxW(0, "The relearning proccess will take a while and will use a lot of proccessing power that may slow your computer\nAre you sure you want the AI to relearn the data?", "Are you sure?", 4)
        if sure == 6:
            gh.set_screen(screen, 3)
            train_size = 0.8
            X_train, Y_train, X_test, Y_test, dic = dp.train_test_vectorization(df, "title", train_size)
            model = lr.learn(X_train, Y_train, X_test, Y_test, dic)
            acc, cf = lr.test(model, X_test, Y_test)
            gh.set_screen(screen, 2)
            print("Relearn process completed:", acc*100, "Percent success rate.")
        else:
            relearn = False
    if not relearn:
        model, X_test, Y_test, dic = lr.load_model()
        acc, cf = lr.test(model, X_test, Y_test)
        gh.set_screen(screen, 1)
        print("Model loaded.", acc*100, "Percent Success rate.")

    present_cf = gh.wait_yes_no()
    if present_cf:
        gh.show_cf(cf)

    gh.set_screen(screen, 4)
    title = gh.wait_text_input(screen, 655, 790, 35, 4, "")
    print(title)
    while title != "0":
        predict, output = lr.guess_one_title(title, model, dic)
        print(predict)
        if predict:
            gh.set_screen(screen, 6)
            title = gh.wait_text_input(screen, 693, 810, 35, 6, output)
            print(title)
        else:
            gh.set_screen(screen, 5)
            title = gh.wait_text_input(screen, 693, 810, 35, 5, output)
            print(title)


start_program()
