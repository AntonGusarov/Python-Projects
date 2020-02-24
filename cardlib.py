import abc, random
from enum import Enum
from collections import Counter


class Suits(Enum):
    """ Ranks the suits and ties them to a value to be able to compare them
    """
    Hearts = 4
    Diamonds = 3
    Spades = 2
    Clubs = 1

class nonnumber(Enum):
    """ Ranks the dressed cards and ties them to a value to be able to compare
    """
    Ace = 14
    King = 13
    Queen = 12
    Jack = 11

class Handvalue(Enum):
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
    """Parent class for all the different card classes

              :param suit: The suit of the card
              :type suit: String

    """
    def __init__(self, suit):
        self.suit = suit

    @abc.abstractmethod
    def get_value(self):
        """ Returns the value of the card
        """
        return self.get_value()

    def __lt__(self, other):
        """Compares if the value of the suit of a card and the value of the card is less then the values for another card
        """
        return (self.suit.value, self.get_value()) < (other.suit.value, other.get_value())

    def __eq__(self, other):
        """Compares if the value of the suit of a card and the value of the card is equal to the values for another card
        """
        return (self.suit.value, self.get_value()) == (other.suit.value, other.get_value())

    def showcard(self):
        """Prints out the name of the card and the name of the suit
        """
        print(self.get_name() + " of " + self.suit.name)


class NumberedCard(PlayingCard):
    """Class for a numbered card with a value between 2-10

              :param val: The value of the card
              :param suit: The suit of the card
              :type val: int
              :type: suit: string

    """
    def __init__(self, val, suit):
        self.value = val
        super().__init__(suit)

    def get_value(self):
        """ Returns the value of the numbered card
        """
        return self.value

    def get_name(self):
        """ Returns the name of the numbered card
        """
        return str(self.value)


class JackCard(PlayingCard):
    """Class for a Jack card

                  :param suit: The suit of the jack card
                  :type: suit: string

    """
    def __init__(self, suit):
        super().__init__(suit)

    def get_value(self):
        """ Returns the value of the Jack card
        """
        return nonnumber.Jack.value

    def get_name(self):
        """ Returns the name of the Jack card
        """
        return nonnumber.Jack.name


class QueenCard(PlayingCard):
    """Class for a Queen card

                      :param suit: The suit of the queen card
                      :type: suit: string

    """
    def __init__(self, suit):
        super().__init__(suit)

    def get_value(self):
        """ Returns the value of the queen card
        """
        return nonnumber.Queen.value

    def get_name(self):
        """ Returns the name of the queen card
        """
        return nonnumber.Queen.name


class KingCard(PlayingCard):
    """Class for a King card

                      :param suit: The suit of the king card
                      :type: suit: string

    """
    def __init__(self, suit):
        super().__init__(suit)

    def get_value(self):
        """ Returns the value of the king card
        """
        return nonnumber.King.value

    def get_name(self):
        """ Returns the name of the king card
        """
        return nonnumber.King.name


class AceCard(PlayingCard):
    """Class for a Ace card

                      :param suit: The suit of the ace card
                      :type: suit: string

    """
    def __init__(self, suit):
        super().__init__(suit)

    def get_value(self):
        """ Returns the value of the ace card
        """
        return nonnumber.Ace.value

    def get_name(self):
        """ Returns the name of the ace card
        """
        return nonnumber.Ace.name

class PokerHand:
    """Class for the poker hand that compares which poker hand has the highest value

              :param poker_hand: type of poker hand
              :type poker_hand: String
              :param highest_values: the highest value in that poker hand, in case two players has the same poker hand
              :type highest_values: tuple

    """
    def __init__(self, poker_hand, highest_values):
        self.poker_hand = Handvalue[poker_hand]
        self.highest_values = highest_values

    def __lt__(self, other):
        """Checks whether self poker hand is valued lower than other poker hand

                   :param other: poker hand object to compare with
                   :type other: poker hand object from PokerHand
                   :return: True if smaller than
                   :rtype: True, False

         """
        return (self.poker_hand.value, self.highest_values) < (self.highest_values, other.poker_hand.value)

    def __eq__(self, other):
        """Checks whether self poker hand is valued equal to other poker hand

                   :param other: poker hand object to compare with
                   :type other: poker hand object from PokerHand
                   :return : True or false
                   :rtype: True, False

         """
        if self.poker_hand.value == other.poker_hand.value:
            return self.highest_values == other.highest_values
        else:
            return False

    def showpokerhand(self):
        print(self.poker_hand.name)

class Hand:
    """Class for the players hand, con pick up and drop cards from the hand
    """
    def __init__(self):
        self.hand = []

    def __lt__(self, other):
        """Compares the value of the card on the hand against a different card on the hand
        """
        return self.hand.get_value() < other.hand.get_value()

    def draw(self, card):
        """Draws a card and adds it to the hand

                  :param card: The card thats added to the hand
                  :type: card: object type PlayingCard

        """
        self.hand.append(card)

    def drop(self, number_of_cards):
        """Draws a card and adds it to the hand

                  :param number_of_cards: Number of cards that is to be dropped
                  :type: number_of_cards: int

        """
        for i in range(number_of_cards):
            self.hand.pop(-1)

    def sorthand(self):
        """Sorts the hand
        """
        self.hand.sort(reverse=True)
        return self.hand

    def best_poker_hand(self, cards=[]):
        """Computes the best poker hand given a hand of cards and a list of cards

                   :param cards: list of cards
                   :type cards: List with card objects of PlayingCard
                   :return: The best poker hand
                   :rtype: poker hand instance of PokerHand

         """
        best_hand = self.evaluate_best_hand(self.hand + cards)
        return PokerHand(best_hand['poker_hand'], best_hand['highest_values'])

    @staticmethod
    def evaluate_best_hand(cards):
        """Evaluates the best possible hand given a list of cards

                    :param cards: A list of playing card objects of PlayingCard.
                    :type cards: Objects of PlayingCard
                    :return: Key value pair with poker_hand as key and highest_values as a tuple
                    :rtype: string, tuple(int)

        """
        if Hand.check_straight_flush(cards):
            return {'poker_hand': 'Straightflush', 'highest_values': Hand.check_straight_flush(cards)}

        elif Hand.check_four_of_a_kind(cards):
            return {'poker_hand': 'Fourofakind', 'highest_values': Hand.check_four_of_a_kind(cards)}

        elif Hand.check_full_house(cards):
            return {'poker_hand': 'Fullhouse', 'highest_values': Hand.check_full_house(cards)}

        elif Hand.check_flush(cards):
            return {'poker_hand': 'Flush', 'highest_values': Hand.check_flush(cards)}

        elif Hand.check_straight(cards):
            return {'poker_hand': 'Straight', 'highest_values': Hand.check_straight(cards)}

        elif Hand.check_three_of_a_kind(cards):
            return {'poker_hand': 'Threeofakind', 'highest_values': Hand.check_three_of_a_kind(cards)}

        elif Hand.check_two_pair(cards):
            return {'poker_hand': 'Twopair', 'highest_values': Hand.check_two_pair(cards)}

        elif Hand.check_one_pair(cards):
            return {'poker_hand': 'Onepair', 'highest_values': Hand.check_one_pair(cards)}

        else:
            max_values = sorted([card.get_value() for card in cards], reverse=True)[:5]
            return {'poker_hand': 'Highcard', 'highest_values': max_values}

    @staticmethod
    def check_straight_flush(cards):
        """ Checks for straight flush in a list of cards

                :param cards: A list of playing cards
                :return: None if no straight flush is found, else highest card of the straight.
                :rtype: int

        """

        vals = [(c.get_value(), c.suit) for c in cards] \
            + [(1, c.suit) for c in cards if c.get_value() == 14]  # Add the aces!
        for c in reversed(cards): # Starting point (high card)
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
                :return: None if no full house is found, else a tuple of the values of the three of a kind and the pair.
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
        """ Checks for four of a kind in a list of cards

                :param cards: A list of playing cards
                :return: None if no four of a kind found, else the value of the four of a kind and value of the kicker
                :rtype: tuple(int, int)

        """

        value_count = Counter()
        for c in cards:
            value_count[c.get_value()] += 1

        value_four_of_a_kind = [v[0] for v in value_count.items() if v[1] == 4]

        # The kicker is chosen as the largest value of the remaining cards
        value_kicker = max([v[0] for v in value_count.items() if v[1] < 4])

        if value_four_of_a_kind:
            return value_four_of_a_kind, value_kicker

    @staticmethod
    def check_flush(cards):
        """ Checks for flush in a list of cards

                :param cards: A list of playing cards
                :return: None if no flush is found, else a tuple of the values of the cards in the flush sorted highest to lowest.
                :rtype: tuple(int, int...int)

        """

        value_count = Counter()
        for c in cards:
            value_count[c.suit] += 1

        suit = [v[0] for v in value_count.items() if v[1] >= 5]

        if suit:
            return sorted([card.get_value() for card in cards if card.suit == suit[0]])
        else:
            return None


    @staticmethod
    def check_straight(cards):
        """ Checks for straight in a list of cards

                 :param cards: A list of playing cards
                 :return: None if no straight is found, else the highest value of the straight.
                 :rtype: int

        """

        vals = [c.get_value() for c in cards] \
            + [1 for c in cards if c.get_value() == 14]  # Add the aces!
        for c in reversed(cards): # Starting point (high card)
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
        """ Checks for three of a kind in a list of cards

                 :param cards: A list of playing cards
                 :return: None if no three of a kind is found, else a tuple of the value of the three of a kind and value of the kicker.
                 :rtype: tuple(int, int)

         """

        value_count = Counter()
        for c in cards:
            value_count[c.get_value()] += 1

        value_three_of_a_kind = [v[0] for v in value_count.items() if v[1] == 3]

        # The two kickers are chosen as the two highest remaining cards
        value_kickers = sorted([v[0] for v in value_count.items() if v[1] == 1], reverse=True)[:2]
        if value_three_of_a_kind:
            return value_three_of_a_kind, value_kickers
        else:
            return None

    @staticmethod
    def check_two_pair(cards):
        """ Checks for two pair in a list of cards

                 :param cards: A list of playing cards
                 :return: None if no two pair is found, else a tuple with the two highest pairs and kicker.
                 :rtype: tuple(int, int, int)

         """

        value_count = Counter()
        for c in cards:
            value_count[c.get_value()] += 1

        value_pairs = [v[0] for v in value_count.items() if v[1] >= 2]

        # The kicker is chosen as the card with highest value of the remaining cards
        value_kicker = max([v[0] for v in value_count.items() if v[1] == 1])

        if len(value_pairs) > 1:
            return sorted(value_pairs, reverse=True)[:2], value_kicker

        else:
            return None

    @staticmethod
    def check_one_pair(cards):
        """ Checks for one pair in a list of cards

                 :param cards: A list of playing cards
                 :return: None if no pair is found, else a tuple with the pair and kickers.
                 :rtype: tuple(int, int)

         """

        value_count = Counter()
        for c in cards:
            value_count[c.get_value()] += 1

        pairs = [v[0] for v in value_count.items() if v[1] >= 2]

        # The kickers are chosen as the cards with highest values of the remaining cards
        values_kickers = sorted([v[0] for v in value_count.items() if v[1] == 1], reverse=True)[:3]

        if len(pairs) > 0:
            return pairs, values_kickers

        else:
            return None


    def showhand(self):
        """A method for showing the cards in the hand
        """
        for card in self.hand:
            card.showcard()


class StandardDeck:
    """CLass of a standard deck of 52 cards
    """
    def __init__(self):
        self.deck = []
        self.suits = [Suits.Hearts, Suits.Diamonds, Suits.Spades, Suits.Clubs]
        for s in self.suits:
            for numbervalue in range(2,11):
                self.deck.append(NumberedCard(numbervalue, s))
            self.deck.append(JackCard(s))
            self.deck.append(QueenCard(s))
            self.deck.append(KingCard(s))
            self.deck.append(AceCard(s))

    def drawfromdeck(self):
        """Method for drawing a card from the deck
        """
        return self.deck.pop(-1)

    def shuffledeck(self):
        """Method for shuffling the deck
        """
        random.shuffle(self.deck)
        return self.deck

    def showdeck(self):
        """Method for showing the whole deck
        """
        for cards in self.deck:
            cards.showcard()

