import pytest

from app.core.prompts import (
    ROLE_RESPONSE_GUIDELINES,
    build_hybrid_prompt_value,
    build_public_customer_prompt_value,
)


@pytest.mark.parametrize(
    "role",
    ["customer", "seller", "support", "operations", "admin"],
)
def test_hybrid_prompt_separates_system_and_human_for_all_roles(role):
    prompt_value = build_hybrid_prompt_value(
        role=role,
        question="Summarize my data",
        structured_context='[{"amount": 1500}]',
        document_context="Sample doc",
        conversation_history="",
    )

    messages = prompt_value.to_messages()
    assert len(messages) == 2
    assert messages[0].type == "system"
    assert messages[1].type == "human"
    assert f"AUTHENTICATED USER ROLE: {role}" in messages[0].content
    assert ROLE_RESPONSE_GUIDELINES[role] in messages[0].content
    assert "Indian Rupees (INR" in messages[0].content
    assert "Summarize my data" in messages[1].content
    assert "Summarize my data" not in messages[0].content


def test_public_customer_prompt_contains_formatting_and_inr_rules():
    prompt_value = build_public_customer_prompt_value(
        question="What is the return policy?",
        context="Returns allowed within 10 calendar days.",
    )

    messages = prompt_value.to_messages()
    assert len(messages) == 2
    assert messages[0].type == "system"
    assert messages[1].type == "human"
    assert "Indian Rupees (INR" in messages[0].content
    assert "RESPONSE FORMATTING RULES" in messages[0].content
    assert "What is the return policy?" in messages[1].content
