import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(16, 8))

layers = [
    ("Input\n48x48x1", (0.1, 0.5)),
    ("Conv2D\n32 (3x3)", (0.3, 0.5)),
    ("Conv2D\n64 (3x3)", (0.5, 0.5)),
    ("MaxPooling\n2x2", (0.7, 0.5)),
    ("Dropout\n25%", (0.9, 0.5)),
    ("Conv2D\n128 (3x3)", (1.1, 0.5)),
    ("MaxPooling\n2x2", (1.3, 0.5)),
    ("Conv2D\n128 (3x3)", (1.5, 0.5)),
    ("MaxPooling\n2x2", (1.7, 0.5)),
    ("Dropout\n25%", (1.9, 0.5)),
    ("Flatten", (2.1, 0.5)),
    ("Dense\n1024", (2.3, 0.5)),
    ("Dropout\n50%", (2.5, 0.5)),
    ("Output\n7", (2.7, 0.5))
]

for layer, (x, y) in layers:
    ax.add_patch(patches.FancyBboxPatch(
        (x, y), 0.1, 0.1, boxstyle="round,pad=0.1", ec="black", fc="lightgray"
    ))
    plt.text(x + 0.05, y + 0.05, layer, ha='center', va='center', fontsize=10)

ax.set_xlim(0, 3)
ax.set_ylim(0, 1)


ax.axis('off')

plt.savefig('cnn_schema.png', bbox_inches='tight')
plt.show()
