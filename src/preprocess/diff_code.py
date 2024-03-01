
patch_text = """@@ -1375,9 +1375,6 @@ public class PiePlot extends Plot implements Cloneable, Serializable {
      * @return The percent.
      */
     public double getMaximumExplodePercent() {
-        if (this.dataset == null) {
-            return 0.0;
-        }
         double result = 0.0;
         Iterator iterator = this.dataset.getKeys().iterator();
         while (iterator.hasNext()) {
@@ -2051,10 +2048,8 @@ public class PiePlot extends Plot implements Cloneable, Serializable {
      
         PiePlotState state = new PiePlotState(info);
         state.setPassesRequired(2);
-        if (this.dataset != null) {
             state.setTotal(DatasetUtilities.calculatePieDatasetTotal(
                     plot.getDataset()));
-        }
         state.setLatestAngle(plot.getStartAngle());
         return state;"""

def extract_code_from_patch_modi(patch:str):
    code_lines = patch.splitlines()
    code_lines_stripped = [line.strip() for line in code_lines]

    patched_code = []
    buggy_code = []

    for line in code_lines_stripped:
        if line.startswith('-') or line.startswith('---'): # buggy line
            buggy_code.append(line)

        elif line.startswith('+') or line.startswith('+++'): # patched line
            patched_code.append(line)

        else:
            patched_code.append(line); buggy_code.append(line)

    
    return patched_code, buggy_code





patched_code, buggy_code = extract_code_from_patch_modi(patch_text)
print(patched_code)


# # Define a function to extract the buggy, patched, and context code
# def extract_code_from_patch(patch_text):
#     # Use regular expressions to find lines starting with '-' and '+'
#     buggy_lines = re.findall(r'^- .*', patch_text, re.MULTILINE)
#     patched_lines = re.findall(r'^\+ .*', patch_text, re.MULTILINE)

#     # Extract the context lines around the changed lines
#     context_lines_before = re.findall(r'^ .*', patch_text, re.MULTILINE)
#     context_lines_after = re.findall(r'^ .*', patch_text, re.MULTILINE)

#     print(context_lines_before)
#     # Join the lines to form the complete code
#     buggy_code = '\n'.join(context_lines_before + buggy_lines + context_lines_after).replace('-', '')
#     patched_code = '\n'.join(context_lines_before + patched_lines + context_lines_after).replace('+', '')

#     return buggy_code.strip(), patched_code.strip()

# # Extract buggy, patched, and context code
# buggy_code, patched_code = extract_code_from_patch(patch_text)

# Print the results
    
# print("Buggy code:")
# print(buggy_code)
# print("\nPatched code:")
# print(patched_code)
