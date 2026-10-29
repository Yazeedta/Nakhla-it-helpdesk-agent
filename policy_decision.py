
import json


def decide_policy(
    query,
    department=None,
    affected_users=None,
    nakhlakey_on_lost_phone=None
):
    """
    Apply initial helpdesk policy rules.
    This function does not create tickets.
    """

    q = query.lower()
    dept = (department or "").lower()

    decision = {
        "policy_refs": [],
        "blocked_actions": [],
        "required_actions": [],
        "needs_clarification": None,
        "check_service_status": False,
        "allow_vpn_troubleshooting": True,
        "proposed_ticket": None
    }

    # Rule 1: SMS and voice MFA codes are forbidden
    is_mfa = any(
        term in q
        for term in ("mfa", "nakhlakey", "2fa", "login code")
    )

    asks_for_sms = any(
        term in q
        for term in ("sms", "voice code", "voice call")
    )

    if is_mfa and asks_for_sms:
        decision["policy_refs"].append("SP-002")
        decision["blocked_actions"].append(
            "Do not offer SMS or voice-call MFA codes."
        )

    # Rule 2: Distinguish a lost phone from one left at home
    phone_at_home = any(
        phrase in q
        for phrase in (
            "forgot my phone at home",
            "left my phone at home",
            "phone is at home"
        )
    )

    phone_lost = (
        "phone" in q
        and any(word in q for word in ("lost", "stolen"))
    )

    if phone_at_home:
        decision["policy_refs"].extend(["SP-002", "KB-007"])
        decision["required_actions"].append(
            "Use a one-time backup code."
        )

    elif phone_lost:
        decision["policy_refs"].append("SP-002")

        if nakhlakey_on_lost_phone is None:
            decision["needs_clarification"] = (
                "Was NakhlaKey installed on the lost phone?"
            )

        elif nakhlakey_on_lost_phone:
            decision["required_actions"].append(
                "Report the security incident within 1 hour."
            )
            decision["proposed_ticket"] = {
                "category": "security",
                "priority": "P1",
                "team": "Security Operations"
            }

    # Rule 3: Finance VPN issues require special handling
    is_vpn = any(
        term in q
        for term in ("vpn", "remote access", "nx-")
    )

    if dept == "finance" and is_vpn:
        decision["policy_refs"].extend(["KB-016", "KB-002"])
        decision["check_service_status"] = True
        decision["allow_vpn_troubleshooting"] = False

        decision["required_actions"].append(
            "Check the service status before proposing a ticket."
        )

        priority = (
            "P1"
            if affected_users is not None and affected_users > 3
            else "P2"
        )

        decision["proposed_ticket"] = {
            "category": "security",
            "priority": priority,
            "team": "Security Operations"
        }

    # Remove duplicate document references
    
    # Rule 4: Engineering VPN escalation
    # Engineering follows the normal VPN troubleshooting policy.
    if dept == "engineering" and is_vpn:
        decision["policy_refs"].extend(
            ["KB-002", "KB-015", "KB-016"]
        )

        decision["check_service_status"] = True
        decision["allow_vpn_troubleshooting"] = True

        decision["required_actions"].append(
            "Follow the applicable KB-002 troubleshooting steps. "
            "Escalate if the problem remains unresolved."
        )

        # Standard single-user incident; major incidents are P1.
        priority = (
            "P1"
            if affected_users is not None and affected_users > 3
            else "P3"
        )

        decision["proposed_ticket"] = {
            "category": "network_access",
            "priority": priority,
            "team": "Network Operations"
        }

    decision["policy_refs"] = list(
        dict.fromkeys(decision["policy_refs"])
    )

    return decision


if __name__ == "__main__":
    tests = [
        {
            "query": "My VPN shows NX-403.",
            "department": "Finance",
            "affected_users": 1
        },
        {
            "query": "I lost my phone. Can I receive an MFA code by SMS?"
        },
        {
            "query": "I lost my phone. Can I receive an MFA code by SMS?",
            "nakhlakey_on_lost_phone": True
        },
        {
            "query": "I left my phone at home. Can I use MFA?"
        }
    ]

    for test in tests:
        print("\n" + "=" * 60)
        print("Question:", test["query"])

        result = decide_policy(**test)

        print(json.dumps(result, indent=2))

