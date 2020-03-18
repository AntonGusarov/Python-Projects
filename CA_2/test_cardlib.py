import cardlib


# Tests < and == functions
def test_lt_eq_card():
    card1 = cardlib.NumberedCard(5, cardlib.Suits.Hearts)
    card2 = cardlib.NumberedCard(5, cardlib.Suits.Hearts)
    card3 = cardlib.AceCard(cardlib.Suits.Spades)

    assert card1 == card2
    assert card1 < card3


def test_add_drop_cards():
    deck = cardlib.StandardDeck()
    test_deck1 = [card.get_value() for card in deck.cards]
    deck.shuffledeck()
    test_deck2 = [card.get_value() for card in deck.cards]
    assert deck.cards != []
    assert test_deck1 != test_deck2  # Tests shuffle function

    test_hand = cardlib.Hand()
    test_hand.add(deck.drawfromdeck())
    test_hand.add(deck.drawfromdeck())

    assert len(test_hand.cards) == 2

    test_hand.drop([0])
    assert len(test_hand.cards) == 1


# Test that best_poker_hand returns a PokerHand object and comparison between two PokerHands work
def test_poker_hand():
    # Creates a poker table with 5 cards
    card1 = cardlib.NumberedCard(5, cardlib.Suits.Hearts)
    card2 = cardlib.NumberedCard(5, cardlib.Suits.Hearts)
    card3 = cardlib.AceCard(cardlib.Suits.Spades)
    card4 = cardlib.KingCard(cardlib.Suits.Clubs)
    card5 = cardlib.QueenCard(cardlib.Suits.Diamonds)

    poker_table = cardlib.Hand()
    poker_table.cards = [card1, card2, card3, card4, card5]

    # Creates two hands with two cards
    hand1 = cardlib.Hand()
    hand2 = cardlib.Hand()

    hand1.cards = [cardlib.NumberedCard(10, cardlib.Suits.Diamonds), cardlib.JackCard(cardlib.Suits.Hearts)]
    hand2.cards = [cardlib.NumberedCard(5, cardlib.Suits.Clubs), cardlib.NumberedCard(4, cardlib.Suits.Diamonds)]


    # Create two pokerhands using the hands cards and the poker_table cards
    pokerhand1 = hand1.best_poker_hand(poker_table.cards)   # Should return poker hand 'Straight'
    pokerhand2 = hand2.best_poker_hand(poker_table.cards)   # Should return poker hand 'Three of a kind'

    assert pokerhand1.poker_hand.name == 'Straight'
    assert pokerhand2.poker_hand.name == 'Threeofakind'

    # Testing < operand for the pokerhand comparison
    assert pokerhand2 < pokerhand1
