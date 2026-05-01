
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np

# Create assessment with questions stacked vertically, larger font

fig = plt.figure(figsize=(14, 50))

# QUESTION 1
ax1 = plt.subplot(10, 1, 1)
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 3)
ax1.axis('off')
ax1.text(0.5, 2.6, 'Question 1', fontsize=18, fontweight='bold')

# Draw 4 equal parts, shade 3
for i in range(4):
    rect = Rectangle((1.5 + i*1.2, 1.2), 1, 0.6, linewidth=2.5, edgecolor='black', facecolor='lightblue' if i < 3 else 'white')
    ax1.add_patch(rect)

ax1.text(0.5, 0.8, 'Which equation shows how to decompose 3/4?', fontsize=13, fontweight='bold')
ax1.text(0.5, 0.3, 'A) 3/4 = 1/4 + 1/4 + 1/4     B) 3/4 = 1/2 + 1/4     C) 3/4 = 2/4 + 1/4     D) 3/4 = 1/4 + 2/4', fontsize=12)

# QUESTION 2
ax2 = plt.subplot(10, 1, 2)
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 3)
ax2.axis('off')
ax2.text(0.5, 2.6, 'Question 2', fontsize=18, fontweight='bold')

# Draw 3 equal parts, shade 2
for i in range(3):
    rect = Rectangle((1.5 + i*1.5, 1.2), 1.3, 0.6, linewidth=2.5, edgecolor='black', facecolor='lightcoral' if i < 2 else 'white')
    ax2.add_patch(rect)

ax2.text(0.5, 0.8, 'Which equation matches this model?', fontsize=13, fontweight='bold')
ax2.text(0.5, 0.3, 'A) 2/3 = 1/3 + 1/3     B) 2/3 = 1/2 + 1/6     C) 2/3 = 1/3 + 2/3     D) 3/3 = 1/3 + 2/3', fontsize=12)

# QUESTION 3
ax3 = plt.subplot(10, 1, 3)
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 3)
ax3.axis('off')
ax3.text(0.5, 2.6, 'Question 3', fontsize=18, fontweight='bold')

# Draw 6 equal parts, shade 4
for i in range(6):
    rect = Rectangle((1.5 + i*0.9, 1.2), 0.8, 0.6, linewidth=2, edgecolor='black', facecolor='lightgreen' if i < 4 else 'white')
    ax3.add_patch(rect)

ax3.text(0.5, 0.8, 'How can 4/6 be decomposed?', fontsize=13, fontweight='bold')
ax3.text(0.5, 0.3, 'A) 4/6 = 1/6 + 1/6 + 1/6 + 1/6     B) 4/6 = 2/6 + 2/6     C) 4/6 = 3/6 + 1/6     D) All of the above', fontsize=12)

# QUESTION 4
ax4 = plt.subplot(10, 1, 4)
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 3)
ax4.axis('off')
ax4.text(0.5, 2.6, 'Question 4', fontsize=18, fontweight='bold')

# Draw 8 equal parts, shade 5
for i in range(8):
    rect = Rectangle((1.5 + i*0.7, 1.2), 0.65, 0.6, linewidth=1.5, edgecolor='black', facecolor='lightyellow' if i < 5 else 'white')
    ax4.add_patch(rect)

ax4.text(0.5, 0.8, 'Which shows 5/8 decomposed into equal parts?', fontsize=13, fontweight='bold')
ax4.text(0.5, 0.3, 'A) 5/8 = 2/8 + 2/8 + 1/8     B) 5/8 = 3/8 + 2/8     C) 5/8 = 1/8 + 1/8 + 1/8 + 1/8 + 1/8     D) All of the above', fontsize=12)

# QUESTION 5
ax5 = plt.subplot(10, 1, 5)
ax5.set_xlim(0, 10)
ax5.set_ylim(0, 3)
ax5.axis('off')
ax5.text(0.5, 2.6, 'Question 5', fontsize=18, fontweight='bold')

# Draw 5 equal parts, shade 3
for i in range(5):
    rect = Rectangle((1.5 + i*1.1, 1.2), 1, 0.6, linewidth=2, edgecolor='black', facecolor='plum' if i < 3 else 'white')
    ax5.add_patch(rect)

ax5.text(0.5, 0.8, 'What does 3/5 equal?', fontsize=13, fontweight='bold')
ax5.text(0.5, 0.3, 'A) 1/5 + 2/5     B) 1/5 + 1/5 + 1/5     C) Both A and B     D) 3/5 cannot be decomposed', fontsize=12)

# QUESTION 6
ax6 = plt.subplot(10, 1, 6)
ax6.set_xlim(0, 10)
ax6.set_ylim(0, 3)
ax6.axis('off')
ax6.text(0.5, 2.6, 'Question 6', fontsize=18, fontweight='bold')

# Draw 8 equal parts, shade 6
for i in range(8):
    rect = Rectangle((1.5 + i*0.7, 1.2), 0.65, 0.6, linewidth=1.5, edgecolor='black', facecolor='lightsteelblue' if i < 6 else 'white')
    ax6.add_patch(rect)

ax6.text(0.5, 0.8, 'Which is one way to decompose 6/8?', fontsize=13, fontweight='bold')
ax6.text(0.5, 0.3, 'A) 6/8 = 4/8 + 2/8     B) 6/8 = 3/8 + 3/8     C) 6/8 = 2/8 + 2/8 + 2/8     D) All of the above', fontsize=12)

# QUESTION 7
ax7 = plt.subplot(10, 1, 7)
ax7.set_xlim(0, 10)
ax7.set_ylim(0, 3)
ax7.axis('off')
ax7.text(0.5, 2.6, 'Question 7', fontsize=18, fontweight='bold')

# Draw 10 equal parts, shade 7
for i in range(10):
    rect = Rectangle((1.5 + i*0.56, 1.2), 0.52, 0.6, linewidth=1.2, edgecolor='black', facecolor='salmon' if i < 7 else 'white')
    ax7.add_patch(rect)

ax7.text(0.5, 0.8, 'How many different ways can 7/10 be decomposed?', fontsize=13, fontweight='bold')
ax7.text(0.5, 0.3, 'A) Only 1 way     B) 2 ways     C) More than 2 ways     D) 7 different ways', fontsize=12)

# QUESTION 8
ax8 = plt.subplot(10, 1, 8)
ax8.set_xlim(0, 10)
ax8.set_ylim(0, 3)
ax8.axis('off')
ax8.text(0.5, 2.6, 'Question 8', fontsize=18, fontweight='bold')

# Draw 6 equal parts, shade 5
for i in range(6):
    rect = Rectangle((1.5 + i*0.9, 1.2), 0.8, 0.6, linewidth=2, edgecolor='black', facecolor='khaki' if i < 5 else 'white')
    ax8.add_patch(rect)

ax8.text(0.5, 0.8, 'Which equation is NOT a decomposition of 5/6?', fontsize=13, fontweight='bold')
ax8.text(0.5, 0.3, 'A) 5/6 = 2/6 + 3/6     B) 5/6 = 1/6 + 1/6 + 1/6 + 1/6 + 1/6     C) 5/6 = 1/6 + 4/6     D) 5/6 = 2/6 + 2/6 + 1/6', fontsize=12)

# QUESTION 9
ax9 = plt.subplot(10, 1, 9)
ax9.set_xlim(0, 10)
ax9.set_ylim(0, 3)
ax9.axis('off')
ax9.text(0.5, 2.6, 'Question 9', fontsize=18, fontweight='bold')

# Draw 5 equal parts, shade 4
for i in range(5):
    rect = Rectangle((1.5 + i*1.1, 1.2), 1, 0.6, linewidth=2, edgecolor='black', facecolor='lightcyan' if i < 4 else 'white')
    ax9.add_patch(rect)

ax9.text(0.5, 0.8, '4/5 = 2/5 + ___. What goes in the blank?', fontsize=13, fontweight='bold')
ax9.text(0.5, 0.3, 'A) 1/5     B) 2/5     C) 3/5     D) 4/5', fontsize=12)

# QUESTION 10
ax10 = plt.subplot(10, 1, 10)
ax10.set_xlim(0, 10)
ax10.set_ylim(0, 3)
ax10.axis('off')
ax10.text(0.5, 2.6, 'Question 10', fontsize=18, fontweight='bold')

# Draw 12 equal parts, shade 7
for i in range(12):
    rect = Rectangle((1.5 + i*0.46, 1.2), 0.42, 0.6, linewidth=1, edgecolor='black', facecolor='lightpink' if i < 7 else 'white')
    ax10.add_patch(rect)

ax10.text(0.5, 0.8, 'Which shows 7/12 decomposed correctly?', fontsize=13, fontweight='bold')
ax10.text(0.5, 0.3, 'A) 7/12 = 3/12 + 4/12     B) 7/12 = 5/12 + 2/12     C) 7/12 = 3/12 + 3/12 + 1/12     D) All of the above', fontsize=12)

plt.tight_layout()
plt.savefig('fraction_decomposition_assessment_vertical.png', dpi=150, bbox_inches='tight')
print("Assessment created successfully with vertical layout!")
plt.show()