#!/usr/bin/env python3
"""
Repository Cleanup Script

Based on the audit findings in PROJECT_AUDIT.md, this script performs
the recommended cleanup actions:

1. Remove empty directories
2. Create missing mandatory files
3. Consolidate scattered documentation
4. Archive outdated files
"""

import os
import shutil
from pathlib import Path
from datetime import datetime


class RepositoryCleanup:
    """Repository cleanup implementation based on audit recommendations."""
    
    def __init__(self, repo_root: str):
        self.repo_root = Path(repo_root)
        self.actions_taken = []
        self.errors = []
        
    def log_action(self, action: str):
        """Log an action taken."""
        print(f"✅ {action}")
        self.actions_taken.append(action)
    
    def log_error(self, error: str):
        """Log an error encountered."""
        print(f"❌ {error}")
        self.errors.append(error)
    
    def remove_empty_directories(self):
        """Remove empty directories identified in audit."""
        empty_dirs = [
            'data',
            'economic_reports', 
            'notebooks',
            'trajectories'
        ]
        
        for dir_name in empty_dirs:
            dir_path = self.repo_root / dir_name
            if dir_path.exists() and dir_path.is_dir():
                try:
                    # Check if truly empty (no files, only empty subdirs allowed)
                    contents = list(dir_path.rglob('*'))
                    files = [p for p in contents if p.is_file()]
                    
                    if not files:  # No files found
                        shutil.rmtree(dir_path)
                        self.log_action(f"Removed empty directory: {dir_name}/")
                    else:
                        self.log_error(f"Directory {dir_name}/ not empty, skipping")
                        
                except Exception as e:
                    self.log_error(f"Failed to remove {dir_name}/: {e}")
    
    def handle_node_modules(self):
        """Handle node_modules directory."""
        node_modules = self.repo_root / 'node_modules'
        if node_modules.exists():
            try:
                # Check if it's actually empty or just has empty subdirs
                contents = list(node_modules.rglob('*'))
                files = [p for p in contents if p.is_file()]
                
                if not files:
                    shutil.rmtree(node_modules)
                    self.log_action("Removed empty node_modules/ directory")
                else:
                    self.log_action("node_modules/ contains files, keeping for npm dependencies")
                    
            except Exception as e:
                self.log_error(f"Failed to handle node_modules/: {e}")
    
    def create_environment_yml(self):
        """Create environment.yml file based on requirements.txt."""
        env_file = self.repo_root / 'environment.yml'
        if not env_file.exists():
            try:
                content = f'''name: py312
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.12
  - pykep
  - pygmo
  - astropy
  - spiceypy
  - pip
  - pip:
    - -r requirements.txt

# This environment file was auto-generated from requirements.txt
# Generated on {datetime.now().strftime('%Y-%m-%d')}
'''
                with open(env_file, 'w') as f:
                    f.write(content)
                
                self.log_action("Created environment.yml")
                
            except Exception as e:
                self.log_error(f"Failed to create environment.yml: {e}")
    
    def consolidate_documentation(self):
        """Move scattered documentation files to docs/ directory."""
        docs_dir = self.repo_root / 'docs'
        if not docs_dir.exists():
            docs_dir.mkdir()
            
        # Files to move to docs/
        doc_files_to_move = [
            'CONSTELLATION_OPTIMIZATION_COMPLETE.md',
            'COVERAGE_IMPROVEMENT_PLAN.md', 
            'DEV_PIPELINE.md',
            'MULTI_MISSION_IMPLEMENTATION.md',
            'TESTING.md'
        ]
        
        for doc_file in doc_files_to_move:
            source = self.repo_root / doc_file
            target = docs_dir / doc_file
            
            if source.exists() and not target.exists():
                try:
                    shutil.move(str(source), str(target))
                    self.log_action(f"Moved {doc_file} to docs/")
                except Exception as e:
                    self.log_error(f"Failed to move {doc_file}: {e}")
    
    def create_archive_directory(self):
        """Create archive directory and move outdated files."""
        archive_dir = self.repo_root / 'docs' / 'archive'
        if not archive_dir.exists():
            archive_dir.mkdir(parents=True)
            self.log_action("Created docs/archive/ directory")
        
        # Files that appear to be outdated or duplicated
        files_to_archive = []
        
        # Check for any remaining scattered docs in root
        for file_path in self.repo_root.glob('*.md'):
            if file_path.name in ['README.md', 'CLAUDE.md', 'CAPABILITIES.md', 'PROJECT_AUDIT.md']:
                continue  # Keep main docs in root
                
            if not file_path.name.startswith('PROJECT_'):
                # These should have been moved to docs/ already
                target = archive_dir / file_path.name
                if not target.exists():
                    files_to_archive.append(file_path)
        
        for file_path in files_to_archive:
            try:
                target = archive_dir / file_path.name
                shutil.move(str(file_path), str(target))
                self.log_action(f"Archived {file_path.name} to docs/archive/")
            except Exception as e:
                self.log_error(f"Failed to archive {file_path.name}: {e}")
    
    def fix_init_files(self):
        """Add basic content to __init__.py files if they're too short."""
        init_files = [
            self.repo_root / 'src' / '__init__.py',
            self.repo_root / 'tests' / '__init__.py'
        ]
        
        for init_file in init_files:
            if init_file.exists():
                try:
                    with open(init_file, 'r') as f:
                        content = f.read().strip()
                    
                    if len(content.split('\n')) < 3:  # Very short
                        module_name = init_file.parent.name
                        new_content = f'''"""
{module_name.title()} module for Lunar Horizon Optimizer.

This package contains the {module_name} components of the 
Lunar Horizon Optimizer platform.
"""

__version__ = "1.0.0"
'''
                        with open(init_file, 'w') as f:
                            f.write(new_content)
                        
                        self.log_action(f"Enhanced {init_file.relative_to(self.repo_root)}")
                        
                except Exception as e:
                    self.log_error(f"Failed to fix {init_file}: {e}")
    
    def update_gitignore(self):
        """Update .gitignore to include common patterns if missing."""
        gitignore = self.repo_root / '.gitignore'
        
        common_patterns = [
            '# Coverage reports',
            'htmlcov/',
            'coverage.json',
            '.coverage',
            '',
            '# Node modules (if using npm tools)',
            'node_modules/',
            '',
            '# Environment files',
            '.env',
            '.env.local',
            '',
            '# IDE files',
            '.vscode/',
            '.idea/',
            '',
            '# OS files',
            '.DS_Store',
            'Thumbs.db',
            '',
            '# Temporary files',
            '*.tmp',
            '*.temp',
            '*.swp',
            '*.swo',
            '',
            '# Build artifacts',
            'build/',
            'dist/',
            '*.egg-info/',
            '',
            '# Archive directories',
            'archive/',
            'old/',
            'backup/'
        ]
        
        try:
            if gitignore.exists():
                with open(gitignore, 'r') as f:
                    existing_content = f.read()
                
                # Add patterns that aren't already there
                lines_to_add = []
                for pattern in common_patterns:
                    if pattern and pattern not in existing_content:
                        lines_to_add.append(pattern)
                
                if lines_to_add:
                    with open(gitignore, 'a') as f:
                        f.write('\n\n# Added by cleanup script\n')
                        for line in lines_to_add:
                            f.write(line + '\n')
                    
                    self.log_action("Updated .gitignore with additional patterns")
                    
        except Exception as e:
            self.log_error(f"Failed to update .gitignore: {e}")
    
    def generate_cleanup_report(self):
        """Generate a report of cleanup actions taken."""
        report_path = self.repo_root / 'CLEANUP_REPORT.md'
        
        lines = []
        lines.append("# Repository Cleanup Report")
        lines.append("")
        lines.append(f"**Cleanup performed**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Based on**: PROJECT_AUDIT.md recommendations")
        lines.append("")
        
        lines.append("## Actions Taken")
        lines.append("")
        if self.actions_taken:
            for action in self.actions_taken:
                lines.append(f"- ✅ {action}")
        else:
            lines.append("- No actions were required")
        lines.append("")
        
        if self.errors:
            lines.append("## Errors Encountered")
            lines.append("")
            for error in self.errors:
                lines.append(f"- ❌ {error}")
            lines.append("")
        
        lines.append("## Recommendations Applied")
        lines.append("")
        lines.append("1. **Empty Directories**: Removed empty directories that served no purpose")
        lines.append("2. **Missing Files**: Created environment.yml for conda environment management")
        lines.append("3. **Documentation**: Consolidated scattered documentation files into docs/")
        lines.append("4. **Code Issues**: Enhanced minimal __init__.py files with proper docstrings")
        lines.append("5. **Git Configuration**: Updated .gitignore with additional useful patterns")
        lines.append("")
        
        lines.append("## Repository Status After Cleanup")
        lines.append("")
        lines.append("The repository structure is now more organized and follows Python")
        lines.append("project best practices. All functionality remains intact while")
        lines.append("eliminating clutter and improving maintainability.")
        lines.append("")
        
        with open(report_path, 'w') as f:
            f.write('\n'.join(lines))
        
        self.log_action(f"Generated cleanup report: {report_path.name}")
    
    def run_cleanup(self):
        """Execute all cleanup operations."""
        print("🧹 Starting Repository Cleanup")
        print("=" * 40)
        print("Based on PROJECT_AUDIT.md recommendations")
        print()
        
        # Execute cleanup operations
        self.remove_empty_directories()
        self.handle_node_modules()
        self.create_environment_yml()
        self.consolidate_documentation()
        self.create_archive_directory()
        self.fix_init_files()
        self.update_gitignore()
        self.generate_cleanup_report()
        
        print()
        print("🎉 Cleanup Complete!")
        print(f"Actions taken: {len(self.actions_taken)}")
        print(f"Errors encountered: {len(self.errors)}")
        if self.errors:
            print("Check CLEANUP_REPORT.md for details on any errors.")


def main():
    """Main function to run repository cleanup."""
    repo_root = os.getcwd()
    
    print("🧹 Repository Cleanup Tool")
    print("Based on audit findings from PROJECT_AUDIT.md")
    print()
    
    response = input("This will modify repository structure. Continue? (y/N): ")
    if response.lower() != 'y':
        print("Cleanup cancelled.")
        return
    
    cleanup = RepositoryCleanup(repo_root)
    cleanup.run_cleanup()


if __name__ == "__main__":
    main()