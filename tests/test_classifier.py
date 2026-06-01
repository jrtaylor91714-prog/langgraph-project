"""Tests for clarity entity detection, intent classification, and routing."""

import pytest

from classifier import classifier


class TestEntityDetection:
    def test_apple_inc_stock_query_detects_entity(self):
        assert classifier.detect_entity("What is the stock price of Apple?") == "apple"

    def test_apple_fruit_still_detects_token_but_routing_differs(self):
        # Entity detection finds the word; routing uses intent to reject company path
        assert classifier.detect_entity("What is the color of the apple?") == "apple"


class TestIntentClassification:
    def test_apple_stock_is_business_intent(self):
        entity = classifier.detect_entity("What is the stock price of Apple?")
        assert classifier.classify_intent("What is the stock price of Apple?", entity) == "stock_price"

    def test_apple_fruit_is_out_of_scope(self):
        entity = classifier.detect_entity("What is the color of the apple?")
        assert classifier.classify_intent("What is the color of the apple?", entity) == "out_of_scope"

    def test_tesla_overview(self):
        entity = classifier.detect_entity("What does Tesla do?")
        assert classifier.classify_intent("What does Tesla do?", entity) == "business_overview"

    def test_competitors_follow_up(self):
        assert classifier.classify_intent("What about competitors?", None) == "competitors"


class TestRoutingDecision:
    def test_apple_stock_routes_to_research(self):
        d = classifier.decide("What is the stock price of Apple?")
        assert d.is_company_research is True
        assert d.company == "apple"
        assert d.intent == "stock_price"
        assert d.out_of_scope_reason is None

    def test_apple_fruit_not_company_research(self):
        d = classifier.decide("What is the color of the apple?")
        assert d.is_company_research is False
        assert d.intent == "out_of_scope"
        assert d.out_of_scope_reason is not None

    def test_tesla_overview_research(self):
        d = classifier.decide("What does Tesla do?")
        assert d.is_company_research is True
        assert d.company == "tesla"

    def test_competitors_reuses_prior_company(self):
        d = classifier.decide("What about competitors?", prior_company="tesla")
        assert d.is_company_research is True
        assert d.company == "tesla"
        assert d.intent == "competitors"

    def test_stock_price_without_company_needs_clarification(self):
        d = classifier.decide("What is the stock price?")
        assert d.needs_clarification is True
        assert d.is_company_research is False
        assert d.intent == "needs_clarification"
        assert "stock price" in (d.clarification_question or "").lower()
