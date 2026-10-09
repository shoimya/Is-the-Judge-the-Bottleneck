"""The experiment's choices: the values the pilot settles and T10 freezes before any Test set run.

    python pipeline/settings.py        (or press Run / Debug on this file)

Every value here can change a reported number, so after the freeze (tag v1-frozen) nothing here changes.
Fixed facts that never change the results, such as download links and folder names, sit at the top of the file
that uses them instead. Each ticket adds its own choices here when it needs them.
"""

# Question sets (T02): how many questions each set has, and the seed that picks them.
PILOT_SIZE = 100
TEST_SIZE = 1000
SAMPLE_SEED = 0


if __name__ == "__main__":
    # A short demo: list every setting and its value.
    print("The experiment's settings:")
    for setting_name, setting_value in list(globals().items()):
        if setting_name.isupper():
            print(f"  {setting_name} = {setting_value}")
