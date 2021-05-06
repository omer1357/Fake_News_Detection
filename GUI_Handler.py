"""
This file is responsible for handling the GUI.
"""

# Imports
import pygame
import numpy as np
import sys
import seaborn as sns
import matplotlib.pyplot as plt


def init_pygame(size=600, caption="Omer's Fake News Detector"):  # Function to initial pygame and the graphic values.
    pygame.init()
    screen = pygame.display.set_mode((size, int(size*1.5)))
    pygame.display.set_caption(caption)
    EMPTY = pygame.image.load(".\Graphics\Empty screen.png")
    screen.blit(EMPTY, (0, 0))
    pygame.display.flip()
    return screen


def set_screen(screen, img):  # Function that set the screen and resolution to a selected screen.
    ASK_RELEARN = pygame.image.load(".\Graphics\Would you like the AI to relearn  the data.png")
    LOADED_ASK_CONFUSION = pygame.image.load(".\Graphics\AI loaded successfully. Do you want to see the confusion matrix.png")
    RELEARNED_ASK_CONFUSION = pygame.image.load(".\Graphics\Data relearned successfully. Do you want to see the confusion matrix.png")
    RELEARNING = pygame.image.load(".\Graphics\Relearning.png")
    ASK_TITLE = pygame.image.load(".\Graphics\Enter a news title to see the AI prediction.png")
    FALSE_ASK_TITLE = pygame.image.load(".\Graphics\The AI think it's FAKE! Enter another news title to see the AI prediction.png")
    TRUE_ASK_TITLE = pygame.image.load(".\Graphics\The AI think it's TRUE! Enter another news title to see the AI prediction.png")
    EMPTY = pygame.image.load(".\Graphics\Empty screen.png")
    GUESS = pygame.image.load(".\Graphics\Do you think it's true.png")
    WRONG_GUESS = pygame.image.load(".\Graphics\Wrong! Do you think it's true.png")
    CORRECT_GUESS = pygame.image.load(".\Graphics\Correct! Do you think it's true.png")
    MENU = pygame.image.load(".\Graphics\Menu.png")
    chooser = {
                    0: ASK_RELEARN,
                    1: LOADED_ASK_CONFUSION,
                    2: RELEARNED_ASK_CONFUSION,
                    3: RELEARNING,
                    4: ASK_TITLE,
                    5: FALSE_ASK_TITLE,
                    6: TRUE_ASK_TITLE,
                    7: EMPTY,
                    8: GUESS,
                    9: WRONG_GUESS,
                    10: CORRECT_GUESS,
                    11: MENU,
            }.get(img, MENU)

    if img == 5 or img == 6 or img == 9 or img == 10:
        if pygame.display.get_surface().get_size()[1] != 1050:
            screen = pygame.display.set_mode((600, 1050))
    else:
        if pygame.display.get_surface().get_size()[1] != 900:
            screen = pygame.display.set_mode((600, 900))
    screen.blit(chooser, (0, 0))
    pygame.display.flip()


def check_quit(event):  # Function that checks if a given event is quit, and handle it if needed.
    if event.type == pygame.QUIT:
        pygame.quit()
        sys.exit()


def check_home(event):  # Function that checks if a given event is a click on the home button.
    if event.type == pygame.MOUSEBUTTONDOWN:
        if 25 < event.pos[1] < 135:
            if 495 < event.pos[0] < 570:
                return True
    return False


def wait_action():
    """"
    Function that wait for the user to chosen option in the MENU, and return 0/1 according to his selection.
    """

    while True:
        for event in pygame.event.get():
            check_quit(event)
            if event.type == pygame.MOUSEBUTTONDOWN:
                if 65 < event.pos[0] < 535:
                    if 535 < event.pos[1] < 650:
                        return 1
                    elif 720 < event.pos[1] < 830:
                        return 0


def wait_yes_no(y, let_home):  # Function that wait for the user to click on yes/no/home buttons (yes/no at given y).
    while True:
        for event in pygame.event.get():
            check_quit(event)
            if let_home:
                if check_home(event):
                    return False, "H"
            if event.type == pygame.MOUSEBUTTONDOWN:
                if y < event.pos[1] < y + 95:
                    if 80 < event.pos[0] < 220:
                        return True
                    elif 380 < event.pos[0] < 795:
                        return False


def wait_text_input(screen, yt, yb, font_size, img, text_box):
    """
    Function that wait for the user to enter text input in a text box.
    Because pygame doesn't have a built-in text box, it's implemented in this function as well.
    The text input box tracks the user keyboard's button clicks and reacts according to it.
    """
    input_box = pygame.Rect(40, yt, 505, 90)
    color_inactive = (0, 0, 0)
    color_active = (194, 0, 0)
    color = color_inactive
    active = False
    text = ''
    multiline = []
    font = pygame.font.Font(None, font_size)

    while True:
        for event in pygame.event.get():
            check_quit(event)
            if check_home(event):
                return False
            if event.type == pygame.MOUSEBUTTONDOWN:
                if input_box.collidepoint(event.pos):
                    active = not active
                else:
                    active = False
                color = color_active if active else color_inactive

                if 220 < event.pos[0] < 380 and yb < event.pos[1] < yb + 70:
                    return "".join(multiline) + text

            if event.type == pygame.KEYDOWN:
                if active:
                    if event.key == pygame.K_RETURN:
                        return "".join(multiline) + text
                    elif event.key == pygame.K_BACKSPACE:
                        if text == "":
                            if multiline:
                                text = multiline[-1]
                                multiline = multiline[:-1]
                                text = text[:-1]
                        else:
                            text = text[:-1]
                    else:
                        text += event.unicode

        set_screen(screen, img)
        show_text_on_box(screen, text_box, font_size, 40, 950)
        txt_surface = font.render(text, True, color)
        if txt_surface.get_width() + 25 > 505:
            multiline.append(text)
            text = ''
        if multiline:
            line = 0
            multiline.append(text)
            for i in multiline:
                txt_surface = font.render(i, True, color)
                screen.blit(txt_surface, (input_box.x + 5, input_box.y + 5 + line*25))
                line += 1
            multiline = multiline[:-1]
        else:
            screen.blit(txt_surface, (input_box.x + 5, input_box.y + 5))

        pygame.draw.rect(screen, color, input_box, 2)
        pygame.display.flip()


def show_text_on_box(screen, out, font_size, x, y):  # Function that show given text on box at a given x, y.
    if out == "":
        return
    multiline = []
    text = ''
    input_box = pygame.Rect(x, y, 505, 75)
    font = pygame.font.Font(None, font_size)
    for i in out.split():
        text += i + " "
        txt_surface = font.render(text, True, (0, 0, 0))
        if txt_surface.get_width() + 25 > 505:
            multiline.append(" ".join(text.split()[:-1]))
            text = text.split()[-1] + " "
    if multiline:
        line = 0
        multiline.append(text)
        for i in multiline:
            txt_surface = font.render(i, True, (0, 0, 0))
            screen.blit(txt_surface, (input_box.x + 5, input_box.y + 5 + line * 25))
            line += 1
    else:
        txt_surface = font.render(out, True, (0, 0, 0))
        screen.blit(txt_surface, (input_box.x + 5, input_box.y + 5))

    pygame.display.flip()


def show_cf(cf):  # Function that and shows the given confusion matrix with labels in a new window.
    percentage = cf / np.sum(cf) * 100
    num_data = ["%.2f" % (percentage[0][0]) + "%\n" + str(cf[0][0]), "%.2f" % (percentage[0][1]) + "%\n" + str(cf[0][1]), "%.2f" % (percentage[1][0]) + "%\n" + str(cf[1][0]), "%.2f" % (percentage[1][1]) + "%\n" + str(cf[1][1])]
    labels = [["True Neg\n" + num_data[0], "False Pos\n" + num_data[1]], ["False Neg\n" + num_data[2], "True Pos\n" + num_data[3]]]
    sns.heatmap(cf, annot=labels, fmt="", cmap='Blues')
    plt.show()
