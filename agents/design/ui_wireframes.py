#!/usr/bin/env python3
"""
UI Wireframe Utility Module

Wireframe generation and layout templates for interface design.
"""

from typing import Dict


class WireframeGenerator:
    """Generate wireframe structures for different page types."""
    
    @staticmethod
    def create_wireframe_structure(page_type: str) -> str:
        """Generate wireframe structure for different page types."""
        wireframes = {
            "landing": """
┌─────────────────────────────────┐
│ Logo    Navigation    CTA       │
├─────────────────────────────────┤
│          Hero Section           │
│       [Main Headline]           │
│       [Sub Headline]            │
│       [Primary CTA]             │
├─────────────────────────────────┤
│     Features Section            │
│  [Feat 1] [Feat 2] [Feat 3]    │
├─────────────────────────────────┤
│          Footer                 │
└─────────────────────────────────┘""",
            "dashboard": """
┌─────────────────────────────────┐
│ Logo    Navigation    Profile   │
├─────┬───────────────────────────┤
│Side │     Main Content          │
│Nav  │   [Stats Cards]           │
│     │   [Chart/Graph]           │
│     │   [Data Table]            │
├─────┴───────────────────────────┤
│          Footer                 │
└─────────────────────────────────┘""",
            "form": """
┌─────────────────────────────────┐
│ Logo    Navigation              │
├─────────────────────────────────┤
│        Form Container           │
│       [Form Title]              │
│       [Input Field 1]           │
│       [Input Field 2]           │
│       [Submit Button]           │
├─────────────────────────────────┤
│          Footer                 │
└─────────────────────────────────┘""",
            "article": """
┌─────────────────────────────────┐
│ Logo    Navigation              │
├─────────────────────────────────┤
│      Article Header             │
│       [Article Title]           │
│       [Author & Date]           │
├─────────────────────────────────┤
│      Article Content            │
│       [Article Body]            │
│       [Related Articles]        │
├─────────────────────────────────┤
│          Footer                 │
└─────────────────────────────────┘""",
            "e-commerce": """
┌─────────────────────────────────┐
│ Logo  Search  Cart  Account     │
├─────────────────────────────────┤
│     Category Navigation         │
├─────┬───────────────────────────┤
│Side │   Product Grid            │
│Nav  │  [Prod] [Prod] [Prod]     │
│     │  [Prod] [Prod] [Prod]     │
├─────┴───────────────────────────┤
│          Footer                 │
└─────────────────────────────────┘"""
        }
        
        return wireframes.get(page_type, wireframes["landing"])
    
    @staticmethod
    def get_available_templates() -> Dict[str, str]:
        """Get list of available wireframe templates."""
        return {
            "landing": "Marketing landing page layout",
            "dashboard": "Admin dashboard with sidebar navigation",
            "form": "Form-focused page layout",
            "article": "Content article layout",
            "e-commerce": "Product listing page layout"
        }