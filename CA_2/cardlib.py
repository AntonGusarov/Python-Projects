
"""
CA 2, Card library that can be used for a variety of different card games, Texas Hold'em for example.
Authors: Martin Gustavsson, Anton Gusarov (Group 26)
"""


import abc
import random
from enum import Enum
from collections import Counter


class Suits(Enum):
    """ Ranks the suits and ties them to a value to be able to compare them
    """
    Hearts = 3
    Diamonds = 2
    Spades = 1
    Clubs = 0


class NonNumber(Enum):
    """ Ranks the dressed cards and ties them to a value to be able to compare
    """
    Ace = 14
    King = 13
    Queen = 12
    Jack = 11


class PokerHandValues(Enum):
    """ Ranks the different possible poker hands and ties them to a value to be able to compare them
    """
    Highcard = 1
    Onepair = 2
    Twopair = 3
    Threeofakind = 4
    Straight = 5
    Flush = 6
    Fullhouse = 7
    Fourofakind = 8
    Straightflush = 9


class PlayingCard:
    """A basic playing card, can be dressed cards or numbered cards.

    :param suit: The suit of the card
    :type suit: Enum Suits

    """
    def __init__(self, suit):
        self.suit = suit

    @abc.abstractmethod
    def get_value(self):
        """ Returns the value of the card
        """
        return self.get_value()

    @abc.abstractmethod
    def get_name(self):
        """ Returns the name of the card
        """
        return self.get_name()

    def __lt__(self, other):
        """Compares if one cards value is less then the value of the other card
        """
        return (self.get_value(), self.suit.value) < (other.get_value(), other.suit.value)

    def __eq__(self, other):
        """Compares if one cards value is equal to the value of the other card
        """
        return (self.get_value(), self.suit.value) == (other.get_value(), other.suit.value)

    def __str__(self):
        """When you print a PlayingCard object this string will print
        """
        return f"{self.get_name()} of {self.suit.name}"

    def showcard(self):
        """Method for printing a playing card
        """
        print(self)


class NumberedCard(PlayingCard):
    """Class for a numbered card with a value between 2-10, inherits from parent class PlayingCard

    :param val: The value of the card
    :type val: int

    :param suit: The suit of the card
    :type: suit: Enum Suits

    """
    def __init__(self, val, suit):
        self.value = val
        super().__init__(suit)

    def get_value(self):
        """ Returns the value of the numbered card
        """
        return self.value

    def get_name(self):
        """ Returns the name of the numbered card as a string
        """
        return str(self.value)


class JackCard(PlayingCard):
    """Class for a Jack card

    :param suit: The suit of the jack card
    :type: suit: Enum Suits

    """
    def __init__(self, suit):
        super().__init__(suit)

    def get_value(self):
        """ Returns the value of the Jack card
        """
        return NonNumber.Jack.value

    def get_name(self):
        """ Returns the name of the Jack card
        """
        return NonNumber.Jack.name


class QueenCard(PlayingCard):
    """Class for a Queen card

    :param suit: The suit of the queen card
    :type: suit: Enum Suits

    """
    def __init__(self, suit):
        super().__init__(suit)

    def get_value(self):
        """ Returns the value of the queen card
        """
        return NonNumber.Queen.value

    def get_name(self):
        """ Returns the name of the queen card
        """
        return NonNumber.Queen.name


class KingCard(PlayingCard):
    """Class for a King card

    :param suit: The suit of the king card
    :type: suit: Enum Suits

    """
    def __init__(self, suit):
        super().__init__(suit)

    def get_value(self):
        """ Returns the value of the king card
        """
        return NonNumber.King.value

    def get_name(self):
        """ Returns the name of the king card
        """
        return NonNumber.King.name


class AceCard(PlayingCard):
    """Class for a Ace card

    :param suit: The suit of the ace card
    :type: suit: Enum Suits

    """
    def __init__(self, suit):
        super().__init__(suit)

    def get_value(self):
        """ Returns the value of the ace card
        """
        return NonNumber.Ace.value

    def get_name(self):
        """ Returns the name of the ace card
        """
        return NonNumber.Ace.name


class PokerHand:
    """A class representing a the different poker hands in a poker game, highcard, one pair, two pair etc.

    :param poker_hand: The type of poker hand, full house for example. Must be be a PokerHandValue key
    :type poker_hand: string

    :param value_and_kickers: A value in some way represent the highest values of the poker hand, for example a straight
            from 5-10 has highest value 10. For comparison between two of the same poker hands.
    :type value_and_kickers: int or tuple
    """
    def __init__(self, poker_hand, value_and_kickers):
        self.poker_hand = PokerHandValues[poker_hand]
        self.value_and_kickers = value_and_kickers

    def __lt__(self, other):
        """Checks whether self PokerHand is valued lower than other PokerHand, by first checking PokerHandValues and
        if equal checks highest_values

        :param other: PokerHand object to compare with
        :type other: PokerHand object

        :return: True if smaller than
        :rtype: bool
        """
        return (self.poker_hand.value, self.value_and_kickers) < (other.poker_hand.value, other.value_and_kickers)

    def __eq__(self, other):
        """Checks whether self PokerHand is valued equal to other PokerHand

        :param other: poker hand object to compare with
        :type other: poker hand object from PokerHand

        :return : True if equal to
        :rtype: bool
        """
        return True if self.poker_hand.value == other.poker_hand.value and \
                       self.value_and_kickers == other.value_and_kickers else False

    def showpokerhand(self):
        print(self.poker_hand.name)


class Hand:
    """Class for a hand, stores cards on hand in a list and can pick up cards and drop cards
    """
    def __init__(self):
        self.cards = []

    def add(self, card):
        """Adds a card to the hand

        :param card: The card that is added to the hand
        :type: card: object type PlayingCard
        """
        self.cards.append(card)

    def drop(self, index=[]):
        """Drops cards from hand according to a list of index, for example index=[2,3] will drop cards in position 2 and
        3 from the hand

        :param index: list of index for which cards to drop
        :type: index: list
        """
        index.sort(reverse=True)
        for i in index:
            self.cards.pop(i)

    def sorthand(self):
        """Sorts the cards in the hand
        """
        self.cards.sort(reverse=True)

    def best_poker_hand(self, cards=[]):
        """Takes the list of cards in Hand and another list of cards and returns the best possible poker hand
        as a PokerHand object

        :param cards: a list of playing cards
        :type cards: List

        :return: The best possible poker hand
        :rtype: PokerHand object
        """
        best_hand = self.evaluate_best_hand(self.cards + cards)
        return PokerHand(best_hand['poker_hand'], best_hand['Value/kickers'])

    @staticmethod
    def evaluate_best_hand(cards):
        """Tests what the best type of poker hand is that can be made using the list of playing cards

        :param cards: A list of PlayingCard objects
        :type cards: PlayingCard object

        :return: A dictionary with type of best poker hand and the highest values for that poker hand, which can be
                the value for the poker hand itself but also have additional kickers, which are used to break ties
        :rtype: dictionary
        """
        if Hand.check_straight_flush(cards):
            return {'poker_hand': 'Straightflush', 'Value/kickers': Hand.check_straight_flush(cards)}

        elif Hand.check_four_of_a_kind(cards):
            return {'poker_hand': 'Fourofakind', 'Value/kickers': Hand.check_four_of_a_kind(cards)}

        elif Hand.check_full_house(cards):
            return {'poker_hand': 'Fullhouse', 'Value/kickers': Hand.check_full_house(cards)}

        elif Hand.check_flush(cards):
            return {'poker_hand': 'Flush', 'Value/kickers': Hand.check_flush(cards)}

        elif Hand.check_straight(cards):
            return {'poker_hand': 'Straight', 'Value/kickers': Hand.check_straight(cards)}

        elif Hand.check_three_of_a_kind(cards):
            return {'poker_hand': 'Threeofakind', 'Value/kickers': Hand.check_three_of_a_kind(cards)}

        elif Hand.check_for_pairs(cards):
            return {'poker_hand': 'Twopair', 'Value/kickers': Hand.check_for_pairs(cards)}

        else:
            # If no poker hand type is found, returns the 5 highest valued cards
            highest_valued_cards = sorted([c.get_value() for c in cards], reverse=True)[:5]
            return {'poker_hand': 'Highcard', 'Value/kickers': highest_valued_cards}

    @staticmethod
    def check_straight_flush(cards):
        """ Checks for straight flush in a list of cards

        :param cards: A list of playing cards
        :type cards: list

        :return: If a straight flush is found it returns the highest value of the straight
        :rtype: int
        """
        vals = [(c.get_value(), c.suit) for c in cards] \
            + [(1, c.suit) for c in cards if c.get_value() == 14]  # Add the aces!
        for c in reversed(cards):  # Starting point (high card)
            # Check if we have the value - k in the set of cards:
            found_straight = True
            for k in range(1, 5):
                if (c.get_value() - k, c.suit) not in vals:
                    found_straight = False
                    break
            if found_straight:
                return c.get_value()

    @staticmethod
    def check_full_house(cards):
        """ Checks for full house in a list of cards

        :param cards: A list of playing cards
        :type: cards: list

        :return: If a full house is found the function returns a tuple of the two values of the full house with the
                three of a kind as the first value
        :rtype: tuple(int, int)
        """
        value_count = Counter()
        for c in cards:
            value_count[c.get_value()] += 1
        # Find the card ranks that have at least three of a kind
        threes = [v[0] for v in value_count.items() if v[1] >= 3]
        threes.sort()
        # Find the card ranks that have at least a pair
        twos = [v[0] for v in value_count.items() if v[1] >= 2]
        twos.sort()

        # Threes are dominant in full house, lets check that value first:
        for three in reversed(threes):
            for two in reversed(twos):
                if two != three:
                    return three, two

    @staticmethod
    def check_four_of_a_kind(cards):
        """ Looks for a four of a kind in a list of cards

        :param cards: A list of playing cards
        :type: cards: list

        :return: If four of the same cards are found function returns a tuple with the value of the four of a kind
                first and second the kicker which is used to break ties.
        :rtype: tuple(int, int)
        """
        value_count = Counter()
        for c in cards:
            value_count[c.get_value()] += 1

        # If there is a four of a kind, value stores the value of the four of a kind
        value = [v[0] for v in value_count.items() if v[1] == 4]

        # A kicker is used to break ties, it's the card with the highest value out of the remaining cards not in four of
        # a kind.
        kicker = max([v[0] for v in value_count.items() if v[1] < 4])

        if value:
            return value, kicker

    @staticmethod
    def check_flush(cards):
        """ Checks if there are at least 5 cards with the same suit in a list of cards

        :param cards: A list of playing cards
        :type: cards: list

        :return: If flush is found, function returns a list with the 5 highest valued cards in the flush
        :rtype: list
        """
        value_count = Counter()
        for c in cards:
            value_count[c.suit] += 1

        # if there are 5 cards or more with the same suit in cards, the suit will be stored in flush_suit
        flush_suit = [v[0] for v in value_count.items() if v[1] >= 5]

        # If a flush found, returns the 5 cards with the highest value in the flush
        if flush_suit:
            flush = sorted([c.get_value() for c in cards if c.suit == flush_suit[0]], reverse=True)
            return flush[:5]

    @staticmethod
    def check_straight(cards):
        """ Checks if there are 5 cards with values in a row (aka a straight) in a list of cards

        :param cards: A list of playing cards
        :type: cards: list

        :return: If straight is found, returns the highest value of the straight
        :rtype: int
        """
        vals = [c.get_value() for c in cards] \
            + [1 for c in cards if c.get_value() == 14]  # Add the aces!
        for c in reversed(cards):  # Starting point (high card)
            # Check if we have the value - k in the set of cards:
            found_straight = True
            for k in range(1, 5):
                if c.get_value() - k not in vals:
                    found_straight = False
                    break
            if found_straight:
                return c.get_value()

    @staticmethod
    def check_three_of_a_kind(cards):
        """ Looks for a three of a kind in a list of cards

        :param cards: A list of playing cards
        :type: cards: list

        :return: If three of the same cards are found function returns a tuple with the value of the three of a kind
                first and second the kickers which are used to break ties.
        :rtype: tuple(int, list)
        """
        value_count = Counter()
        for c in cards:
            value_count[c.get_value()] += 1

        # If there is a three of a kind, value stores the value of the three of a kind
        value = [v[0] for v in value_count.items() if v[1] == 3]

        # Kickers are used to break ties, it's the cards with the highest values out of the remaining cards not in
        # three of a kind.
        kickers = sorted([v[0] for v in value_count.items() if v[1] < 3], reverse=True)
        kickers = kickers[:2]

        if value:
            return value, kickers

    @staticmethod
    def check_for_pairs(cards):
        """ Looks for one or several pairs in a list of cards

         :param cards: A list of playing cards
         :type: cards: list

         :return: If one pair is found the function returns a tuple with the value for the pair and a list of kickers.
                If two or more pairs are found the function returns a tuple with the value for the two highest pairs and
                a kicker.
         :rtype: tuple(int, list) or tuple(list, int)
         """
        value_count = Counter()
        for c in cards:
            value_count[c.get_value()] += 1

        # If there is a pair/pairs, value stores the value of the pair/pairs
        value = sorted([v[0] for v in value_count.items() if v[1] == 2], reverse=True)
        value = value[:2]   # if there are three pairs, only the two highest pairs are stored

        # Kickers are used to break ties, it's the card/cards with the highest value/values out of the cards that is
        # not the highest valued pair/pairs
        kickers = ([v[0] for v in value_count.items() if v[1] not in value])

        if len(value) == 2:
            kicker = max(kickers)
            return value, kicker
        elif len(value) == 1:
            return value, kickers

    def showhand(self):
        """A method for showing the cards in the hand
        """
        for card in self.cards:
            card.showcard()


class StandardDeck:
    """CLass of a standard deck of 52 cards with methods for shuffling the deck and drawing a card from the top of the
    deck and also a method for showing the whole deck.
    """
    def __init__(self):
        self.cards = []
        for s in Suits:
            for number in range(2, 11):
                self.cards.append(NumberedCard(number, s))
            self.cards.append(JackCard(s))
            self.cards.append(QueenCard(s))
            self.cards.append(KingCard(s))
            self.cards.append(AceCard(s))

    def drawfromdeck(self):
        """Method for drawing a card from the deck
        """
        return self.cards.pop(-1)

    def shuffledeck(self):
        """Method for shuffling the deck
        """
        random.shuffle(self.cards)
        return self.cards

    def showdeck(self):
        """Method for showing the whole deck
        """
        for cards in self.cards:
            cards.showcard()
