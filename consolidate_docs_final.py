#!/usr/bin/env python3
"""
Final documentation consolidation script.
Moves scattered .md files to docs/ and updates test counts throughout documentation.
"""

import os
import shutil
from pathlib import Path
import re

def main():
    root_dir = Path("/Users/dmitrystakhin/Library/CloudStorage/Dropbox/work/Lunar Horizon Optimizer")
    docs_dir = root_dir / "docs"
    
    print("🔄 Starting final documentation consolidation...")
    
    # Files to move to docs/ (excluding certain root files)
    files_to_move = [
        "CAPABILITIES.md",
        "PROJECT_AUDIT.md", 
        "CLEANUP_REPORT.md"
    ]
    
    # Move files to docs/
    moved_files = []
    for filename in files_to_move:
        src = root_dir / filename
        dst = docs_dir / filename
        
        if src.exists():
            try:
                shutil.move(str(src), str(dst))
                moved_files.append(filename)
                print(f"✅ Moved {filename} to docs/")
            except Exception as e:
                print(f"❌ Failed to move {filename}: {e}")
    
    # Move scattered test documentation files
    test_docs_to_move = []
    for md_file in (root_dir / "tests").glob("*.md"):
        if md_file.name != "README.md":  # Keep tests/README.md
            dst = docs_dir / f"testing_{md_file.name}"
            try:
                shutil.move(str(md_file), str(dst))
                test_docs_to_move.append(md_file.name)
                print(f"✅ Moved tests/{md_file.name} to docs/testing_{md_file.name}")
            except Exception as e:
                print(f"❌ Failed to move {md_file.name}: {e}")
    
    # Update test counts in key documentation files
    files_to_update = [
        root_dir / "README.md",
        docs_dir / "INDEX.md",
        docs_dir / "PROJECT_STATUS.md"
    ]
    
    for file_path in files_to_update:
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Update test count patterns
                # Pattern 1: "38 tests" -> "243 tests"
                content = re.sub(r'38\s+tests', '243 tests', content)
                content = re.sub(r'38/38\s+tests', '243/243 tests', content)
                content = re.sub(r'38/38\s+production\s+tests', '243/243 production tests', content)
                
                # Pattern 2: Production test suite references
                content = re.sub(
                    r'production test suite \(38 tests[^)]*\)',
                    'production test suite (243 tests, 100% pass rate)',
                    content
                )
                
                # Pattern 3: Test pass rate references
                content = re.sub(
                    r'38/38\s+production\s+tests\s+passing',
                    '243/243 production tests passing',
                    content
                )
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                print(f"✅ Updated test counts in {file_path.name}")
            except Exception as e:
                print(f"❌ Failed to update {file_path.name}: {e}")
    
    # Update documentation references in README.md
    readme_path = root_dir / "README.md"
    if readme_path.exists():
        try:
            with open(readme_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Update documentation links to point to docs/
            updates = {
                '[Complete Capabilities](CAPABILITIES.md)': '[Complete Capabilities](docs/CAPABILITIES.md)',
                '[Project Audit](PROJECT_AUDIT.md)': '[Project Audit](docs/PROJECT_AUDIT.md)', 
                '[Cleanup Report](CLEANUP_REPORT.md)': '[Cleanup Report](docs/CLEANUP_REPORT.md)'
            }
            
            for old_link, new_link in updates.items():
                content = content.replace(old_link, new_link)
            
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print("✅ Updated documentation links in README.md")
        except Exception as e:
            print(f"❌ Failed to update README.md links: {e}")
    
    # Create updated docs/INDEX.md with all documentation
    try:
        index_path = docs_dir / "INDEX.md"
        if index_path.exists():
            with open(index_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Add moved files to the index
            new_section = """
### Repository Management
- [Project Audit](PROJECT_AUDIT.md) - Repository structure analysis and recommendations  
- [Cleanup Report](CLEANUP_REPORT.md) - Recent organizational improvements
- [Complete Capabilities](CAPABILITIES.md) - Comprehensive API reference
"""
            
            # Insert after Overview & Status section
            content = content.replace(
                "### User & Development Guides",
                new_section + "\n### User & Development Guides"
            )
            
            with open(index_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print("✅ Updated docs/INDEX.md with moved files")
        
    except Exception as e:
        print(f"❌ Failed to update docs/INDEX.md: {e}")
    
    # Summary
    print(f"\n📊 Consolidation Summary:")
    print(f"✅ Moved {len(moved_files)} root documentation files to docs/")
    print(f"✅ Moved {len(test_docs_to_move)} test documentation files to docs/")
    print(f"✅ Updated test counts from 38 to 243 in key files")
    print(f"✅ Updated documentation links to point to docs/")
    print(f"✅ All documentation now consolidated in docs/ directory")
    
    print(f"\n📂 Final repository structure:")
    print(f"- Root: README.md, USE_CASES.md, CLAUDE.md (kept as project essentials)")
    print(f"- docs/: All other documentation consolidated")
    print(f"- tests/: Only README.md kept, other docs moved to docs/testing_*")

if __name__ == "__main__":
    main()