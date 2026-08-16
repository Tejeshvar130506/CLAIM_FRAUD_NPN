"""
Role-Based Access Control (RBAC) Module
----------------------------------------
Defines the authorization matrix, permission constants, role hierarchies,
and access control evaluation functions for USER, INVESTIGATOR, MANAGER, and ADMIN roles.
"""

from enum import Enum
from typing import Set, Dict, List, Any


class Role(str, Enum):
    USER = "USER"
    INVESTIGATOR = "INVESTIGATOR"
    MANAGER = "MANAGER"
    ADMIN = "ADMIN"


class Permission(str, Enum):
    # User / Basic Permissions
    UPLOAD_DATASET = "UPLOAD_DATASET"
    EXECUTE_ANALYSIS = "EXECUTE_ANALYSIS"
    VIEW_BASIC_PROVIDER = "VIEW_BASIC_PROVIDER"
    VIEW_OWN_RUNS = "VIEW_OWN_RUNS"
    
    # Investigator Permissions
    VIEW_INVESTIGATION_QUEUE = "VIEW_INVESTIGATION_QUEUE"
    VIEW_DETAILED_EVIDENCE = "VIEW_DETAILED_EVIDENCE"
    VIEW_AGENT_CHAIN = "VIEW_AGENT_CHAIN"
    VIEW_EBM_SHAP_ANALYSIS = "VIEW_EBM_SHAP_ANALYSIS"
    ADD_CASE_NOTE = "ADD_CASE_NOTE"
    RECORD_INVESTIGATION_FINDING = "RECORD_INVESTIGATION_FINDING"
    ESCALATE_TO_MANAGEMENT = "ESCALATE_TO_MANAGEMENT"
    UPDATE_INVESTIGATION_STATUS = "UPDATE_INVESTIGATION_STATUS"
    
    # Manager Permissions
    VIEW_EXECUTIVE_DASHBOARD = "VIEW_EXECUTIVE_DASHBOARD"
    VIEW_ESCALATED_CASES = "VIEW_ESCALATED_CASES"
    RECORD_MANAGEMENT_DECISION = "RECORD_MANAGEMENT_DECISION"
    ASSIGN_INVESTIGATIONS = "ASSIGN_INVESTIGATIONS"
    EXPORT_MANAGEMENT_REPORTS = "EXPORT_MANAGEMENT_REPORTS"
    
    # Admin / System Governance Permissions
    MANAGE_USERS = "MANAGE_USERS"
    MANAGE_ROLES = "MANAGE_ROLES"
    VIEW_AUDIT_LOGS = "VIEW_AUDIT_LOGS"
    VIEW_SYSTEM_HEALTH = "VIEW_SYSTEM_HEALTH"
    MANAGE_MODEL_REGISTRY = "MANAGE_MODEL_REGISTRY"
    CONFIGURE_AGENTS = "CONFIGURE_AGENTS"
    CONFIGURE_RISK_THRESHOLDS = "CONFIGURE_RISK_THRESHOLDS"
    MANAGE_WATCHLIST = "MANAGE_WATCHLIST"


# Permission Assignment Matrix per Role
ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
    Role.USER: {
        Permission.UPLOAD_DATASET,
        Permission.EXECUTE_ANALYSIS,
        Permission.VIEW_BASIC_PROVIDER,
        Permission.VIEW_OWN_RUNS,
    },
    Role.INVESTIGATOR: {
        # Includes User capabilities
        Permission.UPLOAD_DATASET,
        Permission.EXECUTE_ANALYSIS,
        Permission.VIEW_BASIC_PROVIDER,
        Permission.VIEW_OWN_RUNS,
        # Investigator specific
        Permission.VIEW_INVESTIGATION_QUEUE,
        Permission.VIEW_DETAILED_EVIDENCE,
        Permission.VIEW_AGENT_CHAIN,
        Permission.VIEW_EBM_SHAP_ANALYSIS,
        Permission.ADD_CASE_NOTE,
        Permission.RECORD_INVESTIGATION_FINDING,
        Permission.ESCALATE_TO_MANAGEMENT,
        Permission.UPDATE_INVESTIGATION_STATUS,
    },
    Role.MANAGER: {
        # Includes Investigator & User capabilities
        Permission.UPLOAD_DATASET,
        Permission.EXECUTE_ANALYSIS,
        Permission.VIEW_BASIC_PROVIDER,
        Permission.VIEW_OWN_RUNS,
        Permission.VIEW_INVESTIGATION_QUEUE,
        Permission.VIEW_DETAILED_EVIDENCE,
        Permission.VIEW_AGENT_CHAIN,
        Permission.VIEW_EBM_SHAP_ANALYSIS,
        Permission.ADD_CASE_NOTE,
        # Manager specific
        Permission.VIEW_EXECUTIVE_DASHBOARD,
        Permission.VIEW_ESCALATED_CASES,
        Permission.RECORD_MANAGEMENT_DECISION,
        Permission.ASSIGN_INVESTIGATIONS,
        Permission.EXPORT_MANAGEMENT_REPORTS,
    },
    Role.ADMIN: {
        # Admins have all permissions across the entire platform
        *Permission
    }
}


def has_permission(role: str, permission: Permission) -> bool:
    """
    Checks if a given role string has the specified permission.
    """
    try:
        role_enum = Role(role.upper())
        return permission in ROLE_PERMISSIONS.get(role_enum, set())
    except (ValueError, AttributeError):
        return False


def check_permission(role: str, permission: Permission) -> None:
    """
    Raises a PermissionError if the role does not have the specified permission.
    """
    if not has_permission(role, permission):
        raise PermissionError(f"Access Denied: Role '{role}' does not possess permission '{permission.value}'.")


def get_role_permissions(role: str) -> List[str]:
    """Returns a list of permission names for a given role."""
    try:
        role_enum = Role(role.upper())
        return [p.value for p in ROLE_PERMISSIONS.get(role_enum, set())]
    except (ValueError, AttributeError):
        return []


def get_role_dashboard_name(role: str) -> str:
    """Returns human-readable dashboard title for a given role."""
    role_upper = (role or "").upper()
    mapping = {
        "USER": "User Claims & Risk Explorer",
        "INVESTIGATOR": "Special Investigations Unit (SIU) Console",
        "MANAGER": "Executive Fraud Operations Management",
        "ADMIN": "System Administration & Model Governance"
    }
    return mapping.get(role_upper, "Healthcare Intelligence Portal")
