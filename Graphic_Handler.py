import pygame


def init_pygame(size=600, caption="Omer's Fake News Detector"):
    pygame.init()
    screen = pygame.display.set_mode((size, int(size*1.5)))
    pygame.display.set_caption(caption)
    EMPTY = pygame.image.load(".\Graphics\Empty screen.png")
    screen.blit(EMPTY, (0, 0))
    pygame.display.flip()
    return screen


def set_screen(screen, img):
    ASK_RELEARN = pygame.image.load(".\Graphics\Would you like the AI to relearn  the data.png")
    LOADED_ASK_CONFUSION = pygame.image.load(".\Graphics\AI loaded successfully. Do you want to see the confusion matrix.png")
    RELEARNED_ASK_CONFUSION = pygame.image.load(".\Graphics\Data relearned successfully. Do you want to see the confusion matrix.png")
    RELEARNING = pygame.image.load(".\Graphics\Relearning.png")
    ASK_TITLE = pygame.image.load(".\Graphics\Enter a news title to see the AI prediction.png")
    FALSE_ASK_TITLE = pygame.image.load(".\Graphics\The AI think it's FAKE! Enter another news title to see the AI prediction.png")
    TRUE_ASK_TITLE = pygame.image.load(".\Graphics\The AI think it's TRUE! Enter another news title to see the AI prediction.png")
    EMPTY = pygame.image.load(".\Graphics\Empty screen.png")
    chooser = {
                    0: ASK_RELEARN,
                    1: LOADED_ASK_CONFUSION,
                    2: RELEARNED_ASK_CONFUSION,
                    3: RELEARNING,
                    4: ASK_TITLE,
                    5: FALSE_ASK_TITLE,
                    6: TRUE_ASK_TITLE,
                    7: EMPTY,

            }.get(img, EMPTY)
    if img == 5 or img == 6:
        if pygame.display.get_surface().get_size()[1] != 1050:
            screen = pygame.display.set_mode((600, 1050))
    screen.blit(chooser, (0, 0))
    pygame.display.flip()

